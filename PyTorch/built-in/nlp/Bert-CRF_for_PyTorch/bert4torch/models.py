# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from inspect import isfunction
import torch
import torch.nn as nn
import torch_npu
import copy
import time
import json
import re
from bert4torch.layers import LayerNorm, BertEmbeddings, BertLayer, Identity, T5Layer, GatedAttentionUnit, XlnetLayer
from bert4torch.layers import AdaptiveEmbedding, XlnetPositionsEncoding
from bert4torch.snippets import metric_mapping, search_layer, insert_arguments, delete_arguments, get_kw
from bert4torch.snippets import ProgbarLogger, EarlyStopping, FGM, PGD, VAT, IterDataset, take_along_dim
from bert4torch.activations import get_activation
from collections import OrderedDict
import warnings
try:
    from apex import amp
    amp.register_half_function(torch.nn.functional, 'softmax')
    amp.register_half_function(torch, 'fast_gelu')
except ImportError:
    amp = None

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        # 这里主要是为了外面调用用到
        self.global_step, self.local_step, self.total_steps, self.epoch, self.steps_per_epoch, self.train_dataloader = 0, 0, 0, 0, None, None
        self.resume_step, self.resume_epoch = 0, 0
        self.callbacks = []
    
    def save_steps_params(self, save_path):
        '''保存训练过程参数
        '''
        step_params = {'resume_step': (self.local_step+1) % self.steps_per_epoch, 
                       'resume_epoch': self.epoch + (self.local_step+1) // self.steps_per_epoch}
        torch.save(step_params, save_path)

    def load_steps_params(self, save_path):
        '''导入训练过程参数
        '''
        step_params = torch.load(save_path)
        self.resume_step = step_params['resume_step'] 
        self.resume_epoch = step_params['resume_epoch']
        return step_params

    def compile(self, loss, optimizer, scheduler=None, clip_grad_norm=None, use_amp=False, use_apex=True, metrics=None, adversarial_train={'name': ''}):
        '''定义loss, optimizer, metrics, 是否在计算loss前reshape
        loss: loss
        optimizer: 优化器
        scheduler: scheduler
        clip_grad_norm: 是否使用梯度裁剪, 默认不启用
        use_amp: 是否使用混合精度，默认不启用
        metrics: 训练过程中需要打印的指标, loss相关指标默认会打印, 目前支持accuracy, 也支持自定义metric，形式为{key: func}
        '''
        self.criterion = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clip_grad_norm = clip_grad_norm
        self.use_amp = use_amp
        self.use_apex = use_apex

        if use_amp:
            assert adversarial_train['name'] not in {'vat', 'gradient_penalty'}, 'Amp and adversarial_train both run is not supported in current version'
            from torch.cuda.amp import autocast
            self.autocast = autocast
            self.scaler = torch.cuda.amp.GradScaler()

        # 训练过程观测的指标
        self.metrics = OrderedDict({'loss': None})
        self.metrics['step time'] = None
        if metrics is None:
            metrics = []
        elif isinstance(metrics, (str, dict)) or isfunction(metrics):
            metrics = [metrics]

        for metric in metrics:
            # 字符类型，目前仅支持accuracy
            if isinstance(metric, str) and metric != 'loss':
                self.metrics[metric] = None
            # 字典形式 {metric: func}
            elif isinstance(metric, dict):
                self.metrics.update(metric)
            # 函数形式，key和value都赋值metric
            elif isfunction(metric):
                self.metrics.update({metric: metric})
            else:
                raise ValueError('Args metrics only support "String, Dict, Callback, List[String, Dict, Callback]" format')

        # 对抗训练
        self.adversarial = adversarial_train
        self.adversarial_initialize()

    def adversarial_initialize(self):
        '''对抗训练初始化
        '''
        assert self.adversarial['name'] in {'', 'fgm', 'pgd', 'vat', 'gradient_penalty'}, 'adversarial_train support fgm, pgd, vat and gradient_penalty mode'
        self.adversarial['epsilon'] = self.adversarial.get('epsilon', 1.0)
        self.adversarial['emb_name'] = self.adversarial.get('emb_name', 'word_embeddings')

        if self.adversarial['name'] == 'fgm':
            self.ad_train = FGM(self)
        elif self.adversarial['name'] == 'pgd':
            self.adversarial['K'] = self.adversarial.get('K', 3)  # 步数
            self.adversarial['alpha'] = self.adversarial.get('alpha', 0.3)  # 学习率
            self.ad_train = PGD(self)
        elif self.adversarial['name'] == 'gradient_penalty':
            pass
        elif self.adversarial['name'] == 'vat':
            self.adversarial['K'] = self.adversarial.get('K', 3)
            self.adversarial['noise_var'] = self.adversarial.get('noise_var', 1e-5)  # 噪声的方差
            self.adversarial['noise_gamma'] = self.adversarial.get('noise_gamma', 1e-6) # eps
            self.adversarial['adv_step_size'] = self.adversarial.get('adv_step_size', 1e-3)  # 学习率
            self.adversarial['adv_alpha'] = self.adversarial.get('adv_alpha', 1)  # 对抗loss的权重
            self.adversarial['norm_type'] = self.adversarial.get('norm_type', 'l2')  # 归一化方式
            self.ad_train = VAT(self, **self.adversarial)

    def adversarial_training(self, train_X, train_y, output, loss, loss_detail, grad_accumulation_steps):
        '''对抗训练
        '''
        if self.adversarial['name'] == 'fgm':
            self.ad_train.attack(**self.adversarial) # embedding被修改了
            output, loss, loss_detail = self.train_step(train_X, train_y, grad_accumulation_steps)
            loss.backward() # 反向传播，在正常的grad基础上，累加对抗训练的梯度
            # 恢复Embedding的参数, 因为要在正常的embedding上更新参数，而不是增加了对抗扰动后的embedding上更新参数~
            self.ad_train.restore(**self.adversarial)
        elif self.adversarial['name'] == 'pgd':
            self.ad_train.backup_grad()  # 备份梯度
            for t in range(self.adversarial['K']):
                # 在embedding上添加对抗扰动, first attack时备份param.data
                self.ad_train.attack(**self.adversarial, is_first_attack=(t==0))
                if t != self.adversarial['K']-1:
                    self.optimizer.zero_grad()  # 为了累积扰动而不是梯度
                else:
                    self.ad_train.restore_grad() # 恢复正常的grad
                output, loss, loss_detail = self.train_step(train_X, train_y, grad_accumulation_steps)
                loss.backward() # 反向传播，在正常的grad基础上，累加对抗训练的梯度
            self.ad_train.restore(**self.adversarial) # 恢复embedding参数
        # 梯度惩罚
        elif self.adversarial['name'] == 'gradient_penalty':
            para = search_layer(self, self.adversarial['emb_name'], retrun_first=True)
            gp = (para.grad ** 2).sum()
            loss += 0.5 * gp * self.adversarial['epsilon']
            loss.backward()
        # 虚拟对抗训练
        elif self.adversarial['name'] == 'vat':
            logit = output[0] if isinstance(output, (list, tuple)) else output
            adv_loss = self.ad_train.virtual_adversarial_training(train_X, logit)
            loss_detail.update({'loss_sup': loss.item(), 'loss_unsup': adv_loss})
            loss += (adv_loss if adv_loss else 0)
            loss.backward()

        return loss, loss_detail

    def train_step(self, train_X, train_y, grad_accumulation_steps, seq_length=None):
        '''forward并返回loss
        '''
        def args_segmentate(train_X):
            '''参数是否展开
            '''
            if isinstance(train_X, torch.Tensor):  # tensor不展开
                pass
            elif isinstance(self, (BaseModelDP, BaseModelDDP)):
                if self.module.forward.__code__.co_argcount >= 3:
                    return True
            elif self.forward.__code__.co_argcount >= 3:
                return True
            return False

        if self.use_amp:
            with self.autocast():
                output = self.forward(*train_X) if args_segmentate(train_X) else self.forward(train_X)
                loss_detail = self.criterion(output, train_y)
        else:
            output = self.forward(*train_X) if args_segmentate(train_X) else self.forward(train_X)
            loss_detail = self.criterion(output, train_y, seq_length)

        if isinstance(loss_detail, torch.Tensor):
            loss = loss_detail
            loss_detail = {}
        elif isinstance(loss_detail, dict):
            loss = loss_detail['loss']  # 还存在其他loss，仅用于打印
            del loss_detail['loss']
        elif isinstance(loss_detail, (tuple, list)):
            loss = loss_detail[0]
            loss_detail = {f'loss{i}':v for i, v in enumerate(loss_detail[1:], start=1)}
        else:
            raise ValueError('Return loss only support Tensor/dict/tuple/list format')
        # 梯度累积
        loss = loss / grad_accumulation_steps if grad_accumulation_steps > 1 else loss
        return output, loss, loss_detail

    def callback_fun(self, mode, logs={}):
        '''统一调用callback, 方便一些判断条件的触发
        '''
        # 如果是分布式DDP训练，则仅masker_rank可以callback
        if isinstance(self, BaseModelDDP) and self.master_rank!=torch.distributed.get_rank():
            return

        if mode == 'train_begin':
            for callback in self.callbacks:
                callback.on_train_begin()
        elif mode == 'epoch_begin':
            for callback in self.callbacks:
                callback.on_epoch_begin(self.global_step, self.epoch, logs)
        elif mode == 'batch_begin':
            for callback in self.callbacks:
                callback.on_batch_begin(self.global_step, self.local_step, logs)
        elif mode == 'batch_end':
            for callback in self.callbacks:
                callback.on_batch_end(self.global_step, self.local_step, logs)
        elif mode == 'epoch_end':
            for callback in self.callbacks:
                callback.on_epoch_end(self.global_step, self.epoch, logs)
        elif mode == 'train_end':
            for callback in self.callbacks:
                callback.on_train_end()
        elif mode == 'dataloader_end':
            for callback in self.callbacks:
                callback.on_dataloader_end()

    def clip_grad_norm_fused(self, combine_grads, max_norm):
        total_norm = 0
        if max_norm > 0:
            for combine_grad in combine_grads:
                if combine_grad is None:
                    continue
                total_norm += combine_grad.norm(2).pow(2)
            total_norm = total_norm.sqrt()
            clip_coef = max_norm / total_norm.sqrt()
            clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
            for combine_grad in combine_grads:
                if combine_grad is None:
                    continue
                combine_grad.mul_(clip_coef_clamped)
        return total_norm

    def fit(self, train_dataloader, train_sampler, steps_per_epoch=None, epochs=1, grad_accumulation_steps=1, callbacks=None):
        if not hasattr(train_dataloader, '__len__'):
            assert steps_per_epoch is not None, 'Either train_dataloader has attr "__len__" or steps_per_epoch is not None'
        self.steps_per_epoch = len(train_dataloader) if steps_per_epoch is None else steps_per_epoch
        self.total_steps = self.steps_per_epoch * epochs
        self.train_dataloader = train_dataloader  # 设置为成员变量，可由外部的callbacks进行修改
        train_dataloader_iter = iter(self.train_dataloader)  # 循环epoch时不重生成

        callbacks = [] if callbacks is None else callbacks
        callbacks = callbacks if isinstance(callbacks, (list, tuple)) else [callbacks]
        self.callbacks = [ProgbarLogger(epochs, self.steps_per_epoch, [i for i in self.metrics.keys() if isinstance(i, str)])] + callbacks
        self.callback_fun('train_begin')

        # epoch：当前epoch
        # global_step：当前全局训练步数
        # local_step: 当前epoch内的训练步数，不同epoch中相同local_step对应的batch数据不一定相同，在steps_per_epoch=None时相同
        # bti：在dataloader中的index，不同epoch中相同的bti对应的batch数据一般相同，除非重新生成dataloader
        self.bti = 0
        for epoch in range(self.resume_epoch, epochs):
            self.epoch = epoch
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            # resume_step：判断local_step的起点，以及进度条的起始位置
            resume_step = self.resume_step if epoch==self.resume_epoch else 0
            self.callback_fun('epoch_begin')
            start = time.time()
            self.callbacks[0].seen = resume_step
            
            for local_step in range(resume_step, self.steps_per_epoch):
                self.local_step = local_step
                self.global_step = self.epoch * self.steps_per_epoch + self.local_step
                # 循环dataloader, 不要试用itertools的cycle，遇到过变量不释放的问题
                try:
                    batch = next(train_dataloader_iter)
                except StopIteration:
                    self.callback_fun('dataloader_end')  # 适用于数据量较大时，动态读取文件并重新生成dataloader的情况，如预训练
                    train_dataloader_iter = iter(self.train_dataloader)  # shuffle=True时候，其实顺序也重新生成了
                    self.bti = 0
                    batch = next(train_dataloader_iter)
                train_X, train_y, seq_length = batch
                train_X, train_y = train_X.to('npu', non_blocking=True), train_y.to('npu', non_blocking=True)

                # 取btz，最多允许嵌套两层，即((token_ids1, mask1), (token_ids2, mask2))
                # if isinstance(train_X, (list, tuple)):
                #     if isinstance(train_X[0], (list, tuple)):
                #         btz = train_X[0][0].size(0)
                #     else:
                #         btz = train_X[0].size(0)
                # elif isinstance(train_X, torch.Tensor):
                #     btz = train_X.size(0)
                # else:
                #     raise ValueError('Input only support [list, tuple, tensor]')
                # logs = {'batch': self.local_step, 'size': btz}

                logs = OrderedDict()
                self.callback_fun('batch_begin', logs)

                self.train()  # 设置为train模式
                # 入参个数判断，如果入参>=3表示是多个入参，如果=2则表示是一个入参
                output, loss, loss_detail = self.train_step(train_X, train_y, grad_accumulation_steps, seq_length)
                end = time.time()
                step_time = end - start
                start = time.time()
                
                retain_graph = True if self.adversarial['name'] in {'gradient_penalty', 'vat'} else False
                if self.use_amp:  # 混合精度
                    scale_before_step = self.scaler.get_scale()
                    self.scaler.scale(loss).backward(retain_graph=retain_graph)
                else:
                    if self.use_apex:
                        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward(retain_graph=retain_graph)

                # 对抗训练
                loss, loss_detail = self.adversarial_training(train_X, train_y, output, loss, loss_detail, grad_accumulation_steps)
                
                # 参数更新, 真实的参数更新次数要除以grad_accumulation_steps，注意调整总的训练步数
                if (self.global_step+1) % grad_accumulation_steps == 0:
                    skip_scheduler = False
                    # 混合精度
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        if self.clip_grad_norm is not None:  # 梯度裁剪
                            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        skip_scheduler = self.scaler.get_scale() != scale_before_step
                    else:
                        if self.use_apex:
                            if self.clip_grad_norm is not None:  # 梯度裁剪
                                self.clip_grad_norm_fused(self.optimizer.get_optimizer_combined_grads(), self.clip_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_norm)
                        self.optimizer.step()

                    self.optimizer.zero_grad()  # 清梯度
                    if (self.scheduler is not None) and not skip_scheduler:
                        if isinstance(self.scheduler, (tuple, list)):
                            for scheduler in self.scheduler:
                                scheduler.step()
                        else:
                            self.scheduler.step()

                # 添加loss至log打印
                logs.update({'loss': loss.item()})
                if local_step > 5:
                    logs['step time'] = step_time
                logs_loss_detail = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_detail.items()}
                logs.update(logs_loss_detail)
                if self.global_step == resume_step:
                    self.callbacks[0].add_metrics(list(logs_loss_detail.keys()), add_position=1)
                    
                # 添加metrics至log打印
                for metric, func in self.metrics.items():
                    perf = metric_mapping(metric, func, output, train_y)  # 内置的一些accuracy指标
                    if perf is not None:
                        if isfunction(metric):  # 直接传入回调函数(无key)
                            if self.global_step == resume_step:
                                self.callbacks[0].add_metrics(list(perf.keys()))
                            logs.update(perf)
                        elif isinstance(metric, str):  # 直接传入回调函数(有key)
                            logs[metric] = perf
                
                self.callback_fun('batch_end', logs)

                self.bti += 1
            self.callback_fun('epoch_end', logs)
            # earlystop策略
            callback_tmp = [callback_tmp for callback_tmp in self.callbacks if isinstance(callback_tmp, EarlyStopping)]
            if callback_tmp and callback_tmp[0].stopped_epoch > 0:
                break
        self.callback_fun('train_end', logs)

    @torch.no_grad()
    def predict(self, input_tensor_list, return_all=None):
        self.eval()
        if self.forward.__code__.co_argcount >= 3:
            output = self.forward(*input_tensor_list)
        else:
            output = self.forward(input_tensor_list)
        if return_all is None:
            return output
        elif isinstance(output, (tuple, list)) and isinstance(return_all, int) and return_all < len(output):
            return output[return_all]
        else:
            raise ValueError('Return format error')
    
    def load_weights(self, load_path, strict=True, prefix=None):
        '''加载模型权重
           save_path: 权重加载路径
           prefix: None表示按照当前的key加载, 传入string表示按照variable_mapping()中原始的key加载
        '''
        state_dict = torch.load(load_path, map_location='cpu')
        if prefix is None:
            self.load_state_dict(state_dict, strict=strict)
        else:
            # 按照variable_mapping()中原始的key加载
            eval_str = 'self.variable_mapping()' if prefix == '' else f'self.{prefix}.variable_mapping()'
            mapping = {v:k for k, v in eval(eval_str).items()}
            mapping = mapping if prefix == '' else {k:f'{prefix}.{v}' for k,v in mapping.items()}
            state_dict_raw = {}
            for k, v in state_dict.items():
                k = mapping.get(k, k)
                state_dict_raw[k] = v
            self.load_state_dict(state_dict_raw, strict=strict)

    def save_weights(self, save_path, prefix=None):
        '''保存模型权重
           save_path: 权重保存路径
           prefix: None表示按照当前的key加载, 传入string表示按照variable_mapping()中原始的key保存
        '''
        if prefix is None:
            torch.save(self.state_dict(), save_path)
        else:  
            # 按照variable_mapping()中原始的key保存，方便其他官方代码加载模型
            eval_str = 'self.variable_mapping()' if prefix == '' else f'self.{prefix}.variable_mapping()'
            mapping = eval(eval_str)
            mapping = mapping if prefix == '' else {f'{prefix}.{k}':v for k,v in mapping.items()}
            state_dict_raw = {}
            for k, v in self.state_dict().items():
                k = mapping.get(k, k)
                state_dict_raw[k] = v
            torch.save(state_dict_raw, save_path)
    

class BaseModelDP(BaseModel, nn.DataParallel):
    '''DataParallel模式使用多gpu的方法
    '''
    def __init__(self, *args, **kwargs):
        nn.DataParallel.__init__(self, *args, **kwargs)


class BaseModelDDP(BaseModel, nn.parallel.DistributedDataParallel):
    '''DistributedDataParallel模式使用多gpu的方法
    '''
    def __init__(self, *args, master_rank=0, **kwargs):
        self.master_rank = master_rank  # 用于记录打印条的rank
        nn.parallel.DistributedDataParallel.__init__(self, *args, **kwargs)


class BERT_BASE(BaseModel):
    """模型基类
    """

    def __init__(
            self,
            vocab_size,  # 词表大小
            hidden_size,  # 编码维度
            num_hidden_layers,  # Transformer总层数
            num_attention_heads,  # Attention的头数
            intermediate_size,  # FeedForward的隐层维度
            hidden_act,  # FeedForward隐层的激活函数
            dropout_rate=None,  # Dropout比例
            attention_probs_dropout_prob=None,  # Attention矩阵的Dropout比例
            embedding_size=None,  # 指定embedding_size, 不指定则使用config文件的参数
            attention_head_size=None,  # Attention中V的head_size
            attention_key_size=None,  # Attention中Q,K的head_size
            initializer_range=0.02,  # 权重初始化方差
            sequence_length=None,  # 是否固定序列长度
            keep_tokens=None,  # 要保留的词ID列表
            compound_tokens=None,  # 扩展Embedding
            residual_attention_scores=False,  # Attention矩阵加残差
            ignore_invalid_weights=False,  # 允许跳过不存在的权重
            keep_hidden_layers=None, # 保留的hidden_layer层的id
            hierarchical_position=None,  # 是否层次分解位置编码
            **kwargs
    ):
        super(BERT_BASE, self).__init__()
        if keep_tokens is not None:
            vocab_size = len(keep_tokens)
        if compound_tokens is not None:
            vocab_size += len(compound_tokens)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size or self.hidden_size // self.num_attention_heads
        self.attention_key_size = attention_key_size or self.attention_head_size
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout_rate or 0
        self.attention_probs_dropout_prob = attention_probs_dropout_prob or 0
        self.hidden_act = hidden_act
        self.embedding_size = embedding_size or hidden_size
        self.initializer_range = initializer_range
        self.sequence_length = sequence_length
        self.keep_tokens = keep_tokens
        self.compound_tokens = compound_tokens
        self.attention_bias = None
        self.position_bias = None
        self.attention_scores = None
        self.residual_attention_scores = residual_attention_scores
        self.ignore_invalid_weights = ignore_invalid_weights
        self.keep_hidden_layers = set(range(num_hidden_layers)) if keep_hidden_layers is None else set(keep_hidden_layers)
        self.hierarchical_position = hierarchical_position

    def build(
        self,
        attention_caches=None,
        layer_norm_cond=None,
        layer_norm_cond_hidden_size=None,
        layer_norm_cond_hidden_act=None,
        additional_input_layers=None,
        **kwargs
    ):
        """模型构建函数
        attention_caches: 为Attention的K,V的缓存序列字典，格式为{Attention层名: [K缓存, V缓存]}；
        layer_norm_*系列参数: 实现Conditional Layer Normalization时使用，用来实现以“固定长度向量”为条件的条件Bert。
        """
        # additional_input
        # if additional_input_layers is not None:
        #     if not isinstance(additional_input_layers, list):
        #         self.additional_input_layers = [additional_input_layers]
        #     else:
        #         self.additional_input_layers = additional_input_layers

        # Other
        self.attention_caches = attention_caches or {}
        # self.layer_norm_conds = [
        #     layer_norm_cond,
        #     layer_norm_cond_hidden_size,
        #     layer_norm_cond_hidden_act or 'linear',
        # ]
        self.output_all_encoded_layers = kwargs.get('output_all_encoded_layers', False)
        

    def forward(self, inputs):
        """定义模型的执行流程
        """
        # Embedding
        outputs = self.apply_embeddings(inputs)
        # Main
        outputs = self.apply_main_layers(outputs)
        # Final
        outputs = self.apply_final_layers(outputs)
        return outputs

    def init_model_weights(self, module):
        """ 初始化权重
        """
        if isinstance(module, (nn.Linear, nn.Embedding)) and (module.weight.requires_grad):
            # bert参数初始化, tf版本在linear和Embedding层使用的是截断正太分布, pytorch没有实现该函数,
            # 此种初始化对于加载预训练模型后进行finetune没有任何影响，
            # cf https://github.com/pytorch/pytorch/pull/5617
            # 固定的相对位置编码如Sinusoidal无需初始化
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, LayerNorm):
            if hasattr(module, 'bias') and module.bias.requires_grad:  # T5等模型使用的是rmsnorm
                module.bias.data.zero_()
            if hasattr(module, 'weight') and module.weight.requires_grad:
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and (module.bias is not None) and (module.bias.requires_grad):
            module.bias.data.zero_()

    def variable_mapping(self):
        """构建pytorch层与checkpoint的变量名之间的映射表
        """
        return {}

    def load_variable(self):
        raise NotImplementedError

    def load_embeddings(self, embeddings):
        """根据keep_tokens和compound_tokens对embedding进行修改
        """
        if self.keep_tokens is not None:
            embeddings = embeddings[self.keep_tokens]

        if self.compound_tokens is not None:
            ext_embeddings = []
            for item in self.compound_tokens:
                try:
                    ext_embeddings.append(torch.mean(embeddings[item], 0) * torch.ones_like(embeddings[item]))
                except IndexError:
                    ext_embeddings.append(torch.mean(embeddings, 0, keepdim=True))
                    warnings.warn(f'Initialize ext_embeddings from compound_tokens not in embedding index')
            embeddings = torch.cat([embeddings] + ext_embeddings, 0)

        return embeddings

    def load_pos_embeddings(self, embeddings):
        """根据hierarchical_position对pos_embedding进行修改
        """
        if self.hierarchical_position is not None:
            alpha = 0.4 if self.hierarchical_position is True else self.hierarchical_position
            embeddings = embeddings - alpha * embeddings[:1]
            embeddings = embeddings / (1 - alpha)
            position_index = torch.arange(self.max_position)[:, None]
            # 为兼容低版本pytorch没有take_along_dim
            embeddings_x = take_along_dim(embeddings,  torch.div(position_index, embeddings.size(0), rounding_mode='trunc'), dim=0)
            embeddings_y = take_along_dim(embeddings, position_index % embeddings.size(0), dim=0)
            embeddings = alpha * embeddings_x + (1 - alpha) * embeddings_y

        return embeddings

    def load_weights_from_pytorch_checkpoint(self, checkpoint, mapping=None):
        """根据mapping从checkpoint加载权重
        """
        file_state_dict = torch.load(checkpoint, map_location='cpu')  # 加载模型文件
        mapping = mapping or self.variable_mapping()
        parameters_set = set([i[0] for i in self.named_parameters()])  # 可更新的变量
        
        # 如果模型文件和模型结构中同时存在，且不在预设的mapping中，则更新mapping
        # 主要是如为了在外部继承BERT后有其他layer，也能自动从checkpoint中加载进来
        for layer_name in parameters_set:
            if (layer_name in file_state_dict) and (layer_name not in mapping):
                mapping.update({layer_name: layer_name})

        state_dict_new ={}
        for new_key, old_key in mapping.items():
            if new_key not in self.state_dict():
                continue
            elif old_key in file_state_dict: # mapping中包含，且模型结构中有
                state_dict_new[new_key] = self.load_variable(file_state_dict, old_key)
            elif (old_key not in file_state_dict) and (not self.ignore_invalid_weights):
                # mapping中包含，但模型文件中没有
                print(f'[WARNIMG] {old_key} not found in pretrain models')
            if new_key in parameters_set:
                parameters_set.remove(new_key)

        # 未能加载预训练权重的Parameter
        if not self.ignore_invalid_weights:
            for key in parameters_set:
                print(f'[WARNIMG] Parameter {key} not loaded from pretrain models')
        del file_state_dict

        # 将ckpt的权重load到模型结构中
        self.load_state_dict(state_dict_new, strict=False)
    
    # def get_inputs(self):
    #     pass
    
    # def set_inputs(self, inputs, additional_input_layers=None):
    #     """设置input和inputs属性
    #     """
    #     pass

    def apply_embeddings(self, inputs):
        raise NotImplementedError

    def apply_main_layers(self, inputs):
        raise NotImplementedError

    def apply_final_layers(self, inputs):
        raise NotImplementedError
    
    def apply_on_layer_begin(self, l_i, inputs):
        '''新增对layer block输入进行操作的函数
        '''
        return inputs
    
    def apply_on_layer_end(self, l_i, inputs):
        '''新增对layer block输出进行操作的函数
        '''
        return inputs

    def compute_attention_bias(self, inputs=None):
        """定义每一层的Attention Bias
        """
        return self.attention_bias

    def compute_position_bias(self, inputs=None):
        """定义每一层的Position Bias（一般相对位置编码用）
        """
        return self.position_bias

    def set_outputs(self, outputs):
        """设置output和oututs属性
        """
        if not isinstance(outputs, list):
            outputs = [outputs]

        outputs = outputs[:]
        self.outputs = outputs
        if len(outputs) > 1:
            self.output = outputs
        else:
            self.output = outputs[0]


class LM_Mask(object):
    """定义下三角Attention Mask（语言模型用）
    """
    def compute_attention_bias(self, inputs=None):
        """通过idxs序列的比较来得到对应的mask
        """
        seq_len = inputs[0].shape[1]
        attention_bias = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.long, device=inputs[0].device), diagonal=0)
        self.attention_bias = attention_bias.unsqueeze(0).unsqueeze(1)
        return self.attention_bias

def extend_with_language_model(InputModel):
    """添加下三角的Attention Mask（语言模型用）
    """
    class LanguageModel(LM_Mask, InputModel):
        """带下三角Attention Mask的派生模型
        """
        def __init__(self, *args, **kwargs):
            kwargs['with_mlm'] = kwargs.get('with_mlm') or True
            super(LanguageModel, self).__init__(*args, **kwargs)

    return LanguageModel

class UniLM_Mask(object):
    """定义UniLM的Attention Mask（Seq2Seq模型用）
    其中source和target的分区，由segment_ids来表示。
    UniLM: https://arxiv.org/abs/1905.03197
    """
    def compute_attention_bias(self, inputs=None):
        """通过idxs序列的比较来得到对应的mask
        """
        segment_ids = inputs[1]
        attention_bias = torch.cumsum(segment_ids, dim=1)
        attention_bias = (attention_bias.unsqueeze(1)) <= (attention_bias.unsqueeze(2))
        self.attention_bias = attention_bias.unsqueeze(1).long()

        return self.attention_bias

def extend_with_unified_language_model(InputModel):
    """添加UniLM的Attention Mask（Seq2Seq模型用）
    """
    class UnifiedLanguageModel(UniLM_Mask, InputModel):
        """带UniLM的Attention Mask的派生模型
        UniLM: https://arxiv.org/abs/1905.03197
        """
        def __init__(self, *args, **kwargs):
            kwargs['with_mlm'] = kwargs.get('with_mlm') or True
            super(UnifiedLanguageModel, self).__init__(*args, **kwargs)

    return UnifiedLanguageModel


class BERT(BERT_BASE):
    """构建BERT模型
    """

    def __init__(
            self,
            max_position,  # 序列最大长度
            segment_vocab_size=2,  # segment总数目
            with_pool=False,  # 是否包含Pool部分
            with_nsp=False,  # 是否包含NSP部分
            with_mlm=False,  # 是否包含MLM部分
            custom_position_ids=False,  # 是否自行传入位置id
            custom_attention_mask=False, # 是否自行传入attention_mask
            shared_segment_embeddings=False,  # 若True，则segment跟token共用embedding
            layer_norm_cond=None,  # conditional layer_norm
            layer_add_embs=None, # addtional_embeddng, 比如加入词性，音调，word粒度的自定义embedding
            is_dropout=False,
            token_pad_ids=0,  # 默认0是padding ids, 但是注意google的mt5padding不是0
            **kwargs  # 其余参数
    ):
        super(BERT, self).__init__(**kwargs)
        self.max_position = max_position
        self.segment_vocab_size = segment_vocab_size
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm
        self.custom_position_ids = custom_position_ids
        self.custom_attention_mask = custom_attention_mask
        self.shared_segment_embeddings = shared_segment_embeddings
        self.is_dropout = is_dropout
        self.token_pad_ids = token_pad_ids
        if self.with_nsp and not self.with_pool:
            self.with_pool = True
        self.layer_norm_conds = layer_norm_cond
        self.layer_add_embs = layer_add_embs
        self.conditional_size = layer_norm_cond.weight.size(1) if layer_norm_cond is not None else None
        self.embeddings = BertEmbeddings(self.vocab_size, self.embedding_size, self.hidden_size, self.max_position, self.segment_vocab_size, self.shared_segment_embeddings, 
                                         self.dropout_rate, self.conditional_size, **get_kw(BertEmbeddings, kwargs))
        kwargs['max_position'] = self.max_position  # 相对位置编码需要使用    
        layer = BertLayer(self.hidden_size, self.num_attention_heads, self.dropout_rate, self.attention_probs_dropout_prob, self.intermediate_size, self.hidden_act, 
                          is_dropout=self.is_dropout, conditional_size=self.conditional_size, **get_kw(BertLayer, kwargs))
        self.encoderLayer = nn.ModuleList([copy.deepcopy(layer) if layer_id in self.keep_hidden_layers else Identity() for layer_id in range(self.num_hidden_layers)])
        if self.with_pool:
            # Pooler部分（提取CLS向量）
            self.pooler = nn.Linear(self.hidden_size, self.hidden_size)
            self.pooler_activation = nn.Tanh() if self.with_pool is True else get_activation(self.with_pool)
            if self.with_nsp:
                # Next Sentence Prediction部分
                # nsp的输入为pooled_output, 所以with_pool为True是使用nsp的前提条件
                self.nsp = nn.Linear(self.hidden_size, 2)
        else:
            self.pooler = None
            self.pooler_activation = None
        if self.with_mlm:
            self.mlmDense = nn.Linear(self.hidden_size, self.hidden_size)
            self.transform_act_fn = get_activation(self.hidden_act)
            self.mlmLayerNorm = LayerNorm(self.hidden_size, eps=1e-12, conditional_size=self.conditional_size)
            self.mlmDecoder = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
            if kwargs.get('tie_emb_prj_weight') is True:
                self.mlmDecoder.weight = self.embeddings.word_embeddings.weight
            self.mlmBias = nn.Parameter(torch.zeros(self.vocab_size))
            self.mlmDecoder.bias = self.mlmBias
        # 下述继承于BERT的有声明新的参数，在这里初始化不能统一初始化到

    def apply_embeddings(self, inputs):
        """BERT的embedding是token、position、segment三者embedding之和
        默认顺序是token_ids, segment_ids(若有), position_ids(若有), custom_attention_mask(若有), conditional_input(若有)
        """
        assert isinstance(inputs, (list, tuple)), f'Inputs only support list,tuple format but passed {type(inputs)}'

        token_ids = inputs[0]
        index_ = 1
        if self.segment_vocab_size > 0:
            segment_ids = inputs[index_]
            index_ += 1
        else:
            segment_ids = None

        if self.custom_position_ids:  # 暂未使用到，暂保留
            position_ids = inputs[index_]
            index_ += 1
        else:
            position_ids = None
        # 根据token_ids创建一个3D的attention mask矩阵，尺寸为[batch_size, 1, 1, to_seq_length]，
        # 目的是为了适配多头注意力机制，从而能广播到[batch_size, num_heads, from_seq_length, to_seq_length]尺寸
        if self.custom_attention_mask:
            attention_mask = inputs[index_].long().unsqueeze(1).unsqueeze(2)
            index_ += 1
        elif (not token_ids.requires_grad) and (token_ids.dtype in {torch.long, torch.int}): # 正常的token_ids
            attention_mask = (token_ids != self.token_pad_ids).long().unsqueeze(1).unsqueeze(2)  # 默认0为mask_value
            if self.token_pad_ids < 0:
                token_ids = token_ids * attention_mask[:,0,0,:]
        else:  # 自定义word_embedding，目前仅有VAT中使用
            attention_mask = self.attention_mask_cache
        self.attention_mask_cache = attention_mask  # 缓存上次用的attention_mask
        
        self.compute_attention_bias([token_ids, segment_ids])  # 根据lm或者unilm需要对mask做调整
        if self.attention_bias is not None:
            attention_mask = attention_mask * self.attention_bias  # 不可访问padding
            # attention_mask = self.attention_bias  # 可以访问padding

        # pytorch >= 1.5时候会导致StopIteration错误
        # https://github.com/huggingface/transformers/issues/3936
        # https://github.com/huggingface/transformers/issues/4189
        # https://github.com/huggingface/transformers/issues/3936
        try:
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # 兼容fp16
        except StopIteration:
            attention_mask = attention_mask.to(dtype=torch.float32)
        
        # 对mask矩阵中，数值为0的转换成很大的负数，使得不需要attention的位置经过softmax后,分数趋近于0
        # attention_mask = (1.0 - attention_mask) * -10000.0
        # conditional layer_norm
        if self.layer_norm_conds is None:
            conditional_emb = None
        else:
            conditional_emb = self.layer_norm_conds(inputs[index_])
            index_ += 1

        # addtional_embeddng, 比如加入词性，音调，word粒度的自定义embedding
        if isinstance(self.layer_add_embs, nn.Module):  # 单个
            additional_embs = [self.layer_add_embs(inputs[index_])]
            index_ += 1
        elif isinstance(self.layer_add_embs, (tuple, list)):  # 多个
            additional_embs = []
            for layer in self.layer_add_embs:
                assert isinstance(layer, nn.Module), 'Layer_add_embs element should be nn.Module'
                additional_embs.append(layer(inputs[index_]))
                index_ += 1
        else:
            additional_embs = None

        # 进入embedding层
        hidden_states = self.embeddings(token_ids, segment_ids, conditional_emb, additional_embs)
        return [hidden_states, attention_mask, conditional_emb] + inputs[index_:]

    def apply_main_layers(self, inputs):
        """BERT的主体是基于Self-Attention的模块
        顺序:Att --> Add --> LN --> FFN --> Add --> LN
        默认第一个是hidden_states, 第二个是attention_mask, 第三个是conditional_emb
        """
        hidden_states, attention_mask, conditional_emb = inputs[:3]
        if len(inputs[3:]) >= 2:
            encoder_hidden_state, encoder_attention_mask = inputs[3], inputs[4]
        else:
            encoder_hidden_state, encoder_attention_mask = None, None

        encoded_layers = [hidden_states] # 添加embedding的输出
        layer_inputs = [hidden_states, attention_mask, conditional_emb, encoder_hidden_state, encoder_attention_mask]
        for l_i, layer_module in enumerate(self.encoderLayer):
            layer_inputs = self.apply_on_layer_begin(l_i, layer_inputs)
            hidden_states = layer_module(*layer_inputs)
            layer_inputs[0] = hidden_states
            layer_inputs = self.apply_on_layer_end(l_i, layer_inputs)

            if self.output_all_encoded_layers:
                encoded_layers.append(hidden_states)
        if not self.output_all_encoded_layers:
            encoded_layers.append(hidden_states)
        return [encoded_layers, conditional_emb]
    
    def apply_final_layers(self, inputs):
        """根据剩余参数决定输出
        """
        # 获取最后一层隐藏层的输出
        encoded_layers, conditional_emb = inputs
        sequence_output = encoded_layers[-1]
        # 是否取最后一层输出
        if not self.output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        # 是否添加pool层
        if self.with_pool:
            pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0]))
        else:
            pooled_output = None
        # 是否添加nsp
        if self.with_pool and self.with_nsp:
            nsp_scores = self.nsp(pooled_output)
        else:
            nsp_scores = None
        # 是否添加mlm
        if self.with_mlm:
            mlm_hidden_state = self.mlmDense(sequence_output)
            mlm_hidden_state = self.transform_act_fn(mlm_hidden_state)
            mlm_hidden_state = self.mlmLayerNorm((mlm_hidden_state, conditional_emb))
            mlm_scores = self.mlmDecoder(mlm_hidden_state)
            mlm_activation = get_activation('linear' if self.with_mlm is True else self.with_mlm)
            mlm_scores = mlm_activation(mlm_scores)
        else:
            mlm_scores = None
        
        outputs = [value for value in [encoded_layers, pooled_output, mlm_scores, nsp_scores] if value is not None]
        return outputs if len(outputs) > 1 else outputs[0]

    def load_variable(self, state_dict, name, prefix='bert'):
        """加载单个变量的函数
        """
        variable = state_dict[name]
        if name in {
            f'{prefix}.embeddings.word_embeddings.weight',
            'cls.predictions.bias',
            'cls.predictions.decoder.weight',
            'cls.predictions.decoder.bias'
        }:
            return self.load_embeddings(variable)
        elif name == f'{prefix}.embeddings.position_embeddings.weight':
            return self.load_pos_embeddings(variable)
        elif name == 'cls.seq_relationship.weight':
            return variable.T
        else:
            return variable

    def variable_mapping(self, prefix='bert'):
        mapping = {
            'embeddings.word_embeddings.weight': f'{prefix}.embeddings.word_embeddings.weight',
            'embeddings.position_embeddings.weight': f'{prefix}.embeddings.position_embeddings.weight',
            'embeddings.segment_embeddings.weight': f'{prefix}.embeddings.token_type_embeddings.weight',
            'embeddings.layerNorm.weight': f'{prefix}.embeddings.LayerNorm.weight',
            'embeddings.layerNorm.bias': f'{prefix}.embeddings.LayerNorm.bias',
            'pooler.weight': f'{prefix}.pooler.dense.weight',
            'pooler.bias': f'{prefix}.pooler.dense.bias',
            'nsp.weight': 'cls.seq_relationship.weight',
            'nsp.bias': 'cls.seq_relationship.bias',
            'mlmDense.weight': 'cls.predictions.transform.dense.weight',
            'mlmDense.bias': 'cls.predictions.transform.dense.bias',
            'mlmLayerNorm.weight': 'cls.predictions.transform.LayerNorm.weight',
            'mlmLayerNorm.bias': 'cls.predictions.transform.LayerNorm.bias',
            'mlmBias': 'cls.predictions.bias',
            'mlmDecoder.weight': 'cls.predictions.decoder.weight',
            'mlmDecoder.bias': 'cls.predictions.decoder.bias'

        }
        for i in range(self.num_hidden_layers):
            prefix_i = f'{prefix}.encoder.layer.%d.' % i
            mapping.update({f'encoderLayer.{i}.multiHeadAttention.q.weight': prefix_i + 'attention.self.query.weight',
                            f'encoderLayer.{i}.multiHeadAttention.q.bias': prefix_i + 'attention.self.query.bias',
                            f'encoderLayer.{i}.multiHeadAttention.k.weight': prefix_i + 'attention.self.key.weight',
                            f'encoderLayer.{i}.multiHeadAttention.k.bias': prefix_i + 'attention.self.key.bias',
                            f'encoderLayer.{i}.multiHeadAttention.v.weight': prefix_i + 'attention.self.value.weight',
                            f'encoderLayer.{i}.multiHeadAttention.v.bias': prefix_i + 'attention.self.value.bias',
                            f'encoderLayer.{i}.multiHeadAttention.o.weight': prefix_i + 'attention.output.dense.weight',
                            f'encoderLayer.{i}.multiHeadAttention.o.bias': prefix_i + 'attention.output.dense.bias',
                            f'encoderLayer.{i}.layerNorm1.weight': prefix_i + 'attention.output.LayerNorm.weight',
                            f'encoderLayer.{i}.layerNorm1.bias': prefix_i + 'attention.output.LayerNorm.bias',
                            f'encoderLayer.{i}.feedForward.intermediateDense.weight': prefix_i + 'intermediate.dense.weight',
                            f'encoderLayer.{i}.feedForward.intermediateDense.bias': prefix_i + 'intermediate.dense.bias',
                            f'encoderLayer.{i}.feedForward.outputDense.weight': prefix_i + 'output.dense.weight',
                            f'encoderLayer.{i}.feedForward.outputDense.bias': prefix_i + 'output.dense.bias',
                            f'encoderLayer.{i}.layerNorm2.weight': prefix_i + 'output.LayerNorm.weight',
                            f'encoderLayer.{i}.layerNorm2.bias': prefix_i + 'output.LayerNorm.bias'
                            })

        return mapping


class ALBERT(BERT):
    def __init__(self, *args, **kwargs):
        super(ALBERT, self).__init__(*args, **kwargs)
        self.encoderLayer = nn.ModuleList([self.encoderLayer[0]])  # 取上述的第一行

    def apply_main_layers(self, inputs):
        """BERT的主体是基于Self-Attention的模块
        顺序:Att --> Add --> LN --> FFN --> Add --> LN
        """
        hidden_states, attention_mask, conditional_emb = inputs[:3]
        if len(inputs[3:]) >= 2:
            encoder_hidden_state, encoder_attention_mask = inputs[3], inputs[4]
        else:
            encoder_hidden_state, encoder_attention_mask = None, None

        encoded_layers = [hidden_states] # 添加embedding的输出
        layer_inputs = [hidden_states, attention_mask, conditional_emb, encoder_hidden_state, encoder_attention_mask]
        for l_i in range(self.num_hidden_layers):
            layer_inputs = self.apply_on_layer_begin(l_i, layer_inputs)
            hidden_states = self.encoderLayer[0](*layer_inputs)
            layer_inputs[0] = hidden_states
            layer_inputs = self.apply_on_layer_end(l_i, layer_inputs)

            if self.output_all_encoded_layers:
                encoded_layers.append(hidden_states)
        if not self.output_all_encoded_layers:
            encoded_layers.append(hidden_states)
        return [encoded_layers, conditional_emb]

    def variable_mapping(self, prefix='albert'):
        mapping = {
            'embeddings.word_embeddings.weight': f'{prefix}.embeddings.word_embeddings.weight',
            'embeddings.position_embeddings.weight': f'{prefix}.embeddings.position_embeddings.weight',
            'embeddings.segment_embeddings.weight': f'{prefix}.embeddings.token_type_embeddings.weight',
            'embeddings.layerNorm.weight': f'{prefix}.embeddings.LayerNorm.weight',
            'embeddings.layerNorm.bias': f'{prefix}.embeddings.LayerNorm.bias',
            'embeddings.embedding_hidden_mapping_in.weight': f'{prefix}.encoder.embedding_hidden_mapping_in.weight',
            'embeddings.embedding_hidden_mapping_in.bias': f'{prefix}.encoder.embedding_hidden_mapping_in.bias',
            'pooler.weight': f'{prefix}.pooler.weight',
            'pooler.bias': f'{prefix}.pooler.bias',
            'nsp.weight': 'sop_classifier.classifier.weight',  # 用名字nsp来替换sop
            'nsp.bias': 'sop_classifier.classifier.bias',
            'mlmDense.weight': 'predictions.dense.weight',
            'mlmDense.bias': 'predictions.dense.bias',
            'mlmLayerNorm.weight': 'predictions.LayerNorm.weight',
            'mlmLayerNorm.bias': 'predictions.LayerNorm.bias',
            'mlmBias': 'predictions.bias',
            'mlmDecoder.weight': 'predictions.decoder.weight',
            'mlmDecoder.bias': 'predictions.decoder.bias'
        }
        i = 0
        prefix_i = f'{prefix}.encoder.albert_layer_groups.{i}.albert_layers.{i}.'
        mapping.update({f'encoderLayer.{i}.multiHeadAttention.q.weight': prefix_i + 'attention.query.weight',
                        f'encoderLayer.{i}.multiHeadAttention.q.bias': prefix_i + 'attention.query.bias',
                        f'encoderLayer.{i}.multiHeadAttention.k.weight': prefix_i + 'attention.key.weight',
                        f'encoderLayer.{i}.multiHeadAttention.k.bias': prefix_i + 'attention.key.bias',
                        f'encoderLayer.{i}.multiHeadAttention.v.weight': prefix_i + 'attention.value.weight',
                        f'encoderLayer.{i}.multiHeadAttention.v.bias': prefix_i + 'attention.value.bias',
                        f'encoderLayer.{i}.multiHeadAttention.o.weight': prefix_i + 'attention.dense.weight',
                        f'encoderLayer.{i}.multiHeadAttention.o.bias': prefix_i + 'attention.dense.bias',
                        f'encoderLayer.{i}.layerNorm1.weight': prefix_i + 'attention.LayerNorm.weight',
                        f'encoderLayer.{i}.layerNorm1.bias': prefix_i + 'attention.LayerNorm.bias',
                        f'encoderLayer.{i}.feedForward.intermediateDense.weight': prefix_i + 'ffn.weight',
                        f'encoderLayer.{i}.feedForward.intermediateDense.bias': prefix_i + 'ffn.bias',
                        f'encoderLayer.{i}.feedForward.outputDense.weight': prefix_i + 'ffn_output.weight',
                        f'encoderLayer.{i}.feedForward.outputDense.bias': prefix_i + 'ffn_output.bias',
                        f'encoderLayer.{i}.layerNorm2.weight': prefix_i + 'full_layer_layer_norm.weight',
                        f'encoderLayer.{i}.layerNorm2.bias': prefix_i + 'full_layer_layer_norm.bias'
                        })

        return mapping

    def load_variable(self, state_dict, name):
        """加载单个变量的函数
        """
        variable = state_dict[name]
        if name in {
            'albert.embeddings.word_embeddings.weight',
            'predictions.bias',
            'predictions.decoder.weight',
            'predictions.decoder.bias'
        }:
            return self.load_embeddings(variable)
        elif name == 'albert.embeddings.position_embeddings.weight':
            return self.load_pos_embeddings(variable)
        elif name == 'sop_classifier.classifier.weight':
            return variable.T
        else:
            return variable


class ALBERT_Unshared(ALBERT):
    def __init__(self, *args, **kwargs):
        super(ALBERT_Unshared).__init__(*args, **kwargs)
        self.encoderLayer = nn.ModuleList([copy.deepcopy(self.encoderLayer[0]) for _ in range(self.num_hidden_layers)])

    def apply_main_layers(self, inputs):
        """BERT的主体是基于Self-Attention的模块
        顺序:Att --> Add --> LN --> FFN --> Add --> LN
        """
        hidden_states, attention_mask, conditional_emb = inputs
        if len(inputs[3:]) >= 2:
            encoder_hidden_state, encoder_attention_mask = inputs[3], inputs[4]
        else:
            encoder_hidden_state, encoder_attention_mask = None, None

        encoded_layers = [hidden_states] # 添加embedding的输出
        layer_inputs = [hidden_states, attention_mask, conditional_emb, encoder_hidden_state, encoder_attention_mask]
        for i in range(self.num_hidden_layers):
            layer_inputs = self.apply_on_layer_begin(i, layer_inputs)
            hidden_states = self.encoderLayer[i](*layer_inputs)
            layer_inputs[0] = hidden_states
            layer_inputs = self.apply_on_layer_end(i, layer_inputs)

            if self.output_all_encoded_layers:
                encoded_layers.append(hidden_states)
        if not self.output_all_encoded_layers:
            encoded_layers.append(hidden_states)
        return [encoded_layers, conditional_emb]


class NEZHA(BERT):
    """华为推出的NAZHA模型
    链接：https://arxiv.org/abs/1909.00204
    """
    def __init__(self, *args, **kwargs):
        # p_bias来控制embedding阶段无pos_embedding, max_relative_position默认取64
        kwargs.update({'p_bias': 'typical_relative', 'max_relative_position': kwargs.get('max_relative_position', 64)})
        super(NEZHA, self).__init__(*args, **kwargs)


class RoFormer(BERT):
    """旋转式位置编码的BERT模型
    链接：https://kexue.fm/archives/8265
    """
    def __init__(self, *args, **kwargs):
        kwargs.update({'p_bias': 'rotary'})
        super(RoFormer, self).__init__(*args, **kwargs)
    
    def load_variable(self, state_dict, name, prefix='roformer'):
        return super().load_variable(state_dict, name, prefix)

    def variable_mapping(self, prefix='roformer'):
        mapping =  super().variable_mapping(prefix)
        del mapping['embeddings.position_embeddings.weight'] # 没有位置编码
        return mapping


class RoFormerV2(RoFormer):
    """RoFormerV2
    改动：去掉bias，简化Norm，优化初始化等。目前初始化暂时还用的bert的初始化，finetune不受影响
    """
    @delete_arguments('with_pool', 'with_nsp')
    def __init__(self, *args, **kwargs):
        kwargs.update({'p_bias': 'rotary', 'weight': False, 'bias': False, 'norm_mode': 'rmsnorm'})
        super(RoFormerV2, self).__init__(*args, **kwargs)
        if self.with_mlm:
            del self.mlmLayerNorm
            del self.mlmBias
            del self.mlmDense
            self.mlmDecoder.register_parameter('bias', None)

    def variable_mapping(self, prefix='roformer'):
        mapping = super().variable_mapping(prefix)
        mapping_new = {}
        for k, v in mapping.items():
            if (not re.search('bias|layernorm', k.lower())) and (not re.search('bias|layernorm', v.lower())):
                mapping_new[k] = v
        return mapping_new

    def apply_final_layers(self, inputs):
        """根据剩余参数决定输出
        """
        # 获取最后一层隐藏层的输出
        encoded_layers, conditional_emb = inputs
        sequence_output = encoded_layers[-1]
        # 是否取最后一层输出
        if not self.output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        # 是否添加mlm
        if self.with_mlm:
            mlm_scores = self.mlmDecoder(sequence_output)
        else:
            mlm_scores = None
        
        outputs = [value for value in [encoded_layers, mlm_scores] if value is not None]
        return outputs if len(outputs) > 1 else outputs[0]


class GAU_alpha(RoFormerV2):
    def __init__(self, *args, **kwargs):
        kwargs.update({'p_bias': 'rotary', 'weight': False, 'bias': False, 'norm_mode': 'rmsnorm', 'normalization': 'softmax_plus'})
        super().__init__(*args, **kwargs)

        layer = self.GAU_Layer(**kwargs)
        self.encoderLayer = nn.ModuleList([copy.deepcopy(layer) if layer_id in self.keep_hidden_layers else Identity() for layer_id in range(self.num_hidden_layers)])
    
    def load_variable(self, state_dict, name, prefix=''):
        variable = state_dict[name]
        return self.load_embeddings(variable) if name in {'embeddings.word_embeddings.weight', 'mlmDecoder.weight'} else variable

    def variable_mapping(self, prefix=''):
        '''在convert脚本里已经把key转成bert4torch可用的
        '''
        return {k: k for k, _ in self.named_parameters()}

    class GAU_Layer(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.gau = GatedAttentionUnit(**kwargs)
            self.dropout1 = nn.Dropout(kwargs.get('dropout_rate'))
            self.layerNorm1 = LayerNorm(**kwargs)
        def forward(self, hidden_states, attention_mask, conditional_emb=None, encoder_hidden_states=None, encoder_attention_mask=None):
            gau_hidden_states = self.gau(hidden_states, attention_mask)
            hidden_states = hidden_states + self.dropout1(gau_hidden_states)
            hidden_states = self.layerNorm1((hidden_states, conditional_emb))
            return hidden_states

    
class ELECTRA(BERT):
    """Google推出的ELECTRA模型
    链接：https://arxiv.org/abs/2003.10555
    """
    @insert_arguments(with_discriminator=False)
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, max_position, **kwargs):
        super(ELECTRA, self).__init__(max_position, **kwargs)
        if self.with_discriminator:
            self.dense = nn.Linear(self.hidden_size, self.hidden_size)
            self.dense_act = get_activation(self.hidden_act)
            self.dense_prediction = nn.Linear(self.hidden_size, 1)
            self.dense_prediction_act = get_activation('sigmoid') if self.with_discriminator is True else get_activation(self.with_discriminator)

    def apply_final_layers(self, inputs):
        hidden_states = super().apply_final_layers(inputs)  # 仅有hidden_state一项输出
        if self.with_discriminator:
            logit = self.dense_act(self.dense(hidden_states))
            return [hidden_states, self.dense_prediction_act(self.dense_prediction(logit))]
        else:
            return hidden_states

    def load_variable(self, state_dict, name):
        """加载单个变量的函数
        """
        return super().load_variable(state_dict, name, prefix='electra')

    def variable_mapping(self):
        mapping = super(ELECTRA, self).variable_mapping(prefix='electra')
        mapping.update({'dense.weight': 'discriminator_predictions.dense.weight', 
                        'dense.bias': 'discriminator_predictions.dense.bias',
                        'dense_prediction.weight': 'discriminator_predictions.dense_prediction.weight',
                        'dense_prediction.bias': 'discriminator_predictions.dense_prediction.bias'}
                        )
        for del_key in ['pooler.weight', 'pooler.bias', 'nsp.weight', 'nsp.bias', 'mlmDense.weight', 'mlmDense.bias', 
                        'mlmLayerNorm.weight', 'mlmLayerNorm.bias', 'mlmBias', 'mlmDecoder.weight', 'mlmDecoder.bias']:
            del mapping[del_key]

        return mapping


class ERNIE(BERT):
    """百度文心 https://github.com/PaddlePaddle/ERNIE
    """
    def __init__(self, *args, **kwargs):
        super(ERNIE, self).__init__(*args, **kwargs)

    def variable_mapping(self):
        mapping = super(ERNIE, self).variable_mapping(prefix='ernie')
        mapping.update({'mlmDecoder.weight': 'ernie.embeddings.word_embeddings.weight',
                        'mlmDecoder.bias': 'cls.predictions.bias'})
        for k, v in mapping.items():
            if ('LayerNorm.weight' in v) or ('LayerNorm.bias' in v):
                v1 = v.replace('.weight', '.gamma').replace('.bias', '.beta')
                mapping[k] = v1
        for del_key in ['nsp.weight', 'nsp.bias']:
            del mapping[del_key]
        return mapping

    def load_variable(self, state_dict, name, prefix='ernie'):
        return super().load_variable(state_dict, name, prefix=prefix)

class Encoder(BERT):
    def __init__(self, *args, **kwargs):
        kwargs['vocab_size'] = kwargs.get('src_vocab_size', kwargs['vocab_size'])
        super().__init__(*args, **kwargs)
        # encoder需要返回encoder_attention_mask
        self.encoder_attention_mask = None
    
    def forward(self, inputs):
        """因为encoder需要返回encoder_attention_mask，因此这里从新定义一下，多返回一个参数
        """
        # Embedding
        outputs = self.apply_embeddings(inputs)
        encoder_attention_mask = [outputs[1]]
        # Main
        outputs = self.apply_main_layers(outputs)
        # Final
        outputs = self.apply_final_layers(outputs)
        return ([outputs] if isinstance(outputs, torch.Tensor) else outputs) + encoder_attention_mask


class Decoder(LM_Mask, BERT):
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, *args, with_lm=True, tie_emb_prj_weight=False, logit_scale=True, **kwargs):
        kwargs['vocab_size'] = kwargs.get('tgt_vocab_size', kwargs['vocab_size'])
        kwargs['is_decoder'] = True  # 标记是decoder
        super().__init__(*args, **kwargs)
        self.decoderLayer = self.encoderLayer
        del self.encoderLayer
        self.with_lm = with_lm

        # 从hidden_states映射到logit
        if self.with_lm:
            self.final_dense = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
            # decoder底层的embedding和顶层的全连接共享
            # [True]: fudan_bart和uer_t5的t5, [False]: mt5和t5_pegasus
            if tie_emb_prj_weight:
                self.final_dense.weight = self.embeddings.word_embeddings.weight
            if logit_scale:  # T5默认会有logit_scale, bart默认没有，所以bart要传入false
                self.x_logit_scale = (self.hidden_size ** -0.5)
            else:
                self.x_logit_scale = 1.

    def apply_main_layers(self, inputs):
        """Dencoder主体是基于Self-Attention、Cross-Attention的模块
        顺序：Att1 --> Add --> LN --> Att2 --> Add -->  LN --> FFN --> Add --> LN
        """
        hidden_states, attention_mask, conditional_emb, encoder_hidden_state, encoder_attention_mask = inputs[:5]
        decoded_layers = [hidden_states] # 添加embedding的输出
        layer_inputs = [hidden_states, attention_mask, conditional_emb, encoder_hidden_state, encoder_attention_mask]
        for i, layer_module in enumerate(self.decoderLayer):
            layer_inputs = self.apply_on_layer_begin(i, layer_inputs)
            hidden_states = layer_module(*layer_inputs)
            layer_inputs[0] = hidden_states
            layer_inputs = self.apply_on_layer_end(i, layer_inputs)

            if self.output_all_encoded_layers:
                decoded_layers.append(hidden_states)
        if not self.output_all_encoded_layers:
            decoded_layers.append(hidden_states)
        return [decoded_layers, conditional_emb]
    
    def apply_final_layers(self, inputs):
        outputs = []
        hidden_states =  super().apply_final_layers(inputs)  # outputs为decoder顶层的hidden_states [btz, seq_len, hdsz]
        outputs.append(hidden_states)
        if self.with_lm:
            logits = self.final_dense(hidden_states) * self.x_logit_scale # outputs为[btz, seq_len, vocab_size]的logits
            activation = get_activation('linear' if self.with_lm is True else self.with_lm)  # 添加激活，一般是线性激活或softmax
            logits = activation(logits)
            outputs.append(logits)
        return outputs

    def variable_mapping(self, prefix='bert'):
        raw_mapping = super().variable_mapping(prefix)
        mapping = {}
        for k, v in raw_mapping.items():
            mapping[k.replace('encoderLayer', 'decoderLayer')] = v
        # for i in range(self.num_hidden_layers):
        #     prefix_i = f'{prefix}.encoder.layer.%d.' % i
        #     mapping.update({
        #         f'decoderLayer.{i}.crossAttention.q.weight': prefix_i + 'crossattention.self.query.weight',
        #         f'decoderLayer.{i}.crossAttention.q.bias': prefix_i + 'crossattention.self.query.bias',
        #         f'decoderLayer.{i}.crossAttention.k.weight': prefix_i + 'crossattention.self.key.weight',
        #         f'decoderLayer.{i}.crossAttention.k.bias': prefix_i + 'crossattention.self.key.bias',
        #         f'decoderLayer.{i}.crossAttention.v.weight': prefix_i + 'crossattention.self.value.weight',
        #         f'decoderLayer.{i}.crossAttention.v.bias': prefix_i + 'crossattention.self.value.bias',
        #         f'decoderLayer.{i}.crossAttention.o.weight': prefix_i + 'crossattention.output.dense.weight',
        #         f'decoderLayer.{i}.crossAttention.o.bias': prefix_i + 'crossattention.output.dense.bias',
        #         f'decoderLayer.{i}.layerNorm3.weight': prefix_i + 'crossattention.output.LayerNorm.weight',
        #         f'decoderLayer.{i}.layerNorm3.bias': prefix_i + 'crossattention.output.LayerNorm.bias'
        #         })
        return mapping

class Transformer(BERT_BASE):
    '''encoder-decoder结构
    '''
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, *args, tie_emb_src_tgt_weight=False, **kwargs):
        super(Transformer, self).__init__(*args, **kwargs)

        # encoder
        self.encoder = Encoder(*args, **kwargs)
        self.encoder.build(**kwargs)

        # decoder
        self.decoder = Decoder(*args, **kwargs)
        self.decoder.build(**kwargs)

        if tie_emb_src_tgt_weight:
            # encoder和decoder的embedding权重共享
            assert self.encoder.vocab_size == self.decoder.vocab_size, "To share word embedding, the vocab size of src/tgt shall be the same."
            self.encoder.embeddings.word_embeddings.weight = self.decoder.embeddings.word_embeddings.weight

    def forward(self, inputs):
        """定义模型的执行流程
        """
        encoder_input, decoder_input = inputs[:2]

        # encoder
        # encoder_emb = self.encoder.apply_embeddings(encoder_input)
        # encode_outputs = self.encoder.apply_main_layers(encoder_emb)
        # encoder_hidden_state = self.encoder.apply_final_layers(encode_outputs)
        # encoder_attention_mask = encoder_emb[1]
        encoder_hidden_state, encoder_attention_mask = self.encoder(encoder_input)

        # decoder
        # decoder_emb = self.decoder.apply_embeddings(decoder_input)
        # decoder_outputs = self.decoder.apply_main_layers([*decoder_emb, encoder_hidden_state, encoder_attention_mask])
        # decoder_outputs = self.decoder.apply_final_layers(decoder_outputs) # [hidden_states, logits]
        decoder_outputs = self.decoder(decoder_input + [encoder_hidden_state, encoder_attention_mask])
        return [encoder_hidden_state] + decoder_outputs  # 输出encoder_hidden_state和decoder_hidden_state，以应对一些多任务情况


class BART(Transformer):
    '''encoder-decoder结构
    '''
    def __init__(self, *args, tie_emb_src_tgt_weight=True, **kwargs):
        kwargs['logit_scale'] = kwargs.get('logit_scale', False)
        kwargs['tie_emb_prj_weight'] = kwargs.get('tie_emb_prj_weight', True)
        super(BART, self).__init__(*args, tie_emb_src_tgt_weight=tie_emb_src_tgt_weight, **kwargs)
        self.tie_emb_src_tgt_weight = tie_emb_src_tgt_weight

    def load_variable(self, state_dict, name, prefix=''):
        """加载单个变量的函数
        """
        variable = state_dict[name]
        if name in {
            'shared.weight',
            'encoder.embed_tokens.weight',
            'decoder.embed_tokens.weight',
        }:
            return self.load_embeddings(variable)
        elif name in {'encoder.embed_positions.weight', 'decoder.embed_positions.weight'}:
            return self.load_pos_embeddings(variable)
        else:
            return variable

    def variable_mapping(self, prefix=''):
        # 查看check_point发现'shared.weight'
        mapping = {
            'encoder.embeddings.word_embeddings.weight': 'shared.weight' if self.tie_emb_src_tgt_weight else 'encoder.embed_tokens.weight',
            'encoder.embeddings.position_embeddings.weight': 'encoder.embed_positions.weight',
            'encoder.embeddings.layerNorm.weight': 'encoder.layernorm_embedding.weight',
            'encoder.embeddings.layerNorm.bias': 'encoder.layernorm_embedding.bias',
            'decoder.embeddings.word_embeddings.weight': 'shared.weight' if self.tie_emb_src_tgt_weight else 'decoder.embed_tokens.weight',
            'decoder.embeddings.position_embeddings.weight': 'decoder.embed_positions.weight',
            'decoder.embeddings.layerNorm.weight': 'decoder.layernorm_embedding.weight',
            'decoder.embeddings.layerNorm.bias': 'decoder.layernorm_embedding.bias',
        }
        for i in range(self.num_hidden_layers):
            mapping.update(
                {
                f'encoder.encoderLayer.{i}.multiHeadAttention.q.weight': f'encoder.layers.{i}.self_attn.q_proj.weight',
                f'encoder.encoderLayer.{i}.multiHeadAttention.q.bias': f'encoder.layers.{i}.self_attn.q_proj.bias',
                f'encoder.encoderLayer.{i}.multiHeadAttention.k.weight': f'encoder.layers.{i}.self_attn.k_proj.weight',
                f'encoder.encoderLayer.{i}.multiHeadAttention.k.bias': f'encoder.layers.{i}.self_attn.k_proj.bias',
                f'encoder.encoderLayer.{i}.multiHeadAttention.v.weight': f'encoder.layers.{i}.self_attn.v_proj.weight',
                f'encoder.encoderLayer.{i}.multiHeadAttention.v.bias': f'encoder.layers.{i}.self_attn.v_proj.bias',
                f'encoder.encoderLayer.{i}.multiHeadAttention.o.weight': f'encoder.layers.{i}.self_attn.out_proj.weight',
                f'encoder.encoderLayer.{i}.multiHeadAttention.o.bias': f'encoder.layers.{i}.self_attn.out_proj.bias',
                f'encoder.encoderLayer.{i}.layerNorm1.weight': f'encoder.layers.{i}.self_attn_layer_norm.weight',
                f'encoder.encoderLayer.{i}.layerNorm1.bias': f'encoder.layers.{i}.self_attn_layer_norm.bias',
                f'encoder.encoderLayer.{i}.feedForward.intermediateDense.weight': f'encoder.layers.{i}.fc1.weight',
                f'encoder.encoderLayer.{i}.feedForward.intermediateDense.bias': f'encoder.layers.{i}.fc1.bias',
                f'encoder.encoderLayer.{i}.feedForward.outputDense.weight': f'encoder.layers.{i}.fc2.weight',
                f'encoder.encoderLayer.{i}.feedForward.outputDense.bias': f'encoder.layers.{i}.fc2.bias',
                f'encoder.encoderLayer.{i}.layerNorm2.weight': f'encoder.layers.{i}.final_layer_norm.weight',
                f'encoder.encoderLayer.{i}.layerNorm2.bias': f'encoder.layers.{i}.final_layer_norm.bias',
                f'decoder.decoderLayer.{i}.multiHeadAttention.q.weight': f'decoder.layers.{i}.self_attn.q_proj.weight',
                f'decoder.decoderLayer.{i}.multiHeadAttention.q.bias': f'decoder.layers.{i}.self_attn.q_proj.bias',
                f'decoder.decoderLayer.{i}.multiHeadAttention.k.weight': f'decoder.layers.{i}.self_attn.k_proj.weight',
                f'decoder.decoderLayer.{i}.multiHeadAttention.k.bias': f'decoder.layers.{i}.self_attn.k_proj.bias',
                f'decoder.decoderLayer.{i}.multiHeadAttention.v.weight': f'decoder.layers.{i}.self_attn.v_proj.weight',
                f'decoder.decoderLayer.{i}.multiHeadAttention.v.bias': f'decoder.layers.{i}.self_attn.v_proj.bias',
                f'decoder.decoderLayer.{i}.multiHeadAttention.o.weight': f'decoder.layers.{i}.self_attn.out_proj.weight',
                f'decoder.decoderLayer.{i}.multiHeadAttention.o.bias': f'decoder.layers.{i}.self_attn.out_proj.bias',
                f'decoder.decoderLayer.{i}.layerNorm1.weight': f'decoder.layers.{i}.self_attn_layer_norm.weight',
                f'decoder.decoderLayer.{i}.layerNorm1.bias': f'decoder.layers.{i}.self_attn_layer_norm.bias',
                f'decoder.decoderLayer.{i}.crossAttention.q.weight': f'decoder.layers.{i}.encoder_attn.q_proj.weight',
                f'decoder.decoderLayer.{i}.crossAttention.q.bias': f'decoder.layers.{i}.encoder_attn.q_proj.bias',
                f'decoder.decoderLayer.{i}.crossAttention.k.weight': f'decoder.layers.{i}.encoder_attn.k_proj.weight',
                f'decoder.decoderLayer.{i}.crossAttention.k.bias': f'decoder.layers.{i}.encoder_attn.k_proj.bias',
                f'decoder.decoderLayer.{i}.crossAttention.v.weight': f'decoder.layers.{i}.encoder_attn.v_proj.weight',
                f'decoder.decoderLayer.{i}.crossAttention.v.bias': f'decoder.layers.{i}.encoder_attn.v_proj.bias',
                f'decoder.decoderLayer.{i}.crossAttention.o.weight': f'decoder.layers.{i}.encoder_attn.out_proj.weight',
                f'decoder.decoderLayer.{i}.crossAttention.o.bias': f'decoder.layers.{i}.encoder_attn.out_proj.bias',
                f'decoder.decoderLayer.{i}.layerNorm3.weight': f'decoder.layers.{i}.encoder_attn_layer_norm.weight',
                f'decoder.decoderLayer.{i}.layerNorm3.bias': f'decoder.layers.{i}.encoder_attn_layer_norm.bias',
                f'decoder.decoderLayer.{i}.feedForward.intermediateDense.weight': f'decoder.layers.{i}.fc1.weight',
                f'decoder.decoderLayer.{i}.feedForward.intermediateDense.bias': f'decoder.layers.{i}.fc1.bias',
                f'decoder.decoderLayer.{i}.feedForward.outputDense.weight': f'decoder.layers.{i}.fc2.weight',
                f'decoder.decoderLayer.{i}.feedForward.outputDense.bias': f'decoder.layers.{i}.fc2.bias',
                f'decoder.decoderLayer.{i}.layerNorm2.weight': f'decoder.layers.{i}.final_layer_norm.weight',
                f'decoder.decoderLayer.{i}.layerNorm2.bias': f'decoder.layers.{i}.final_layer_norm.bias'
                })

        return mapping


class T5_Encoder(Encoder):
    @insert_arguments(version='t5.1.0')
    def __init__(self, *args, **kwargs):
        kwargs.update({'p_bias': 't5_relative', 'relative_attention_num_buckets': kwargs.get('relative_attention_num_buckets'), 'version': self.version, 
                       'bias': False, 'norm_mode': 'rmsnorm'})  # p_bias来控制embedding阶段无pos_embedding，t5不使用bias，并且使用rmsnorm
        super().__init__(*args, **kwargs)
        del self.embeddings.layerNorm

        # t5的layernorm都在前面，因此重新定义了下
        layer = T5Layer(self.hidden_size, self.num_attention_heads, self.dropout_rate, self.attention_probs_dropout_prob, self.intermediate_size, self.hidden_act, is_dropout=self.is_dropout, 
                            conditional_size=self.conditional_size, **get_kw(BertLayer, kwargs))
        self.encoderLayer = nn.ModuleList([copy.deepcopy(layer) for _ in range(self.num_hidden_layers)])

        # 把第二层后的相对位置编码的权重绑定到第一层上，变相实现仅由第一层计算
        for i in range(1, self.num_hidden_layers):
            self.encoderLayer[i].multiHeadAttention.relative_positions_encoding.weight = self.encoderLayer[0].multiHeadAttention.relative_positions_encoding.weight
        self.final_layer_norm = LayerNorm(self.hidden_size, eps=1e-12, conditional_size=self.conditional_size, bias=False, norm_mode='rmsnorm')
        self.dropout = nn.Dropout(self.dropout_rate)

    def apply_final_layers(self, inputs):
        hidden_states = super().apply_final_layers(inputs)
        return self.dropout(self.final_layer_norm([hidden_states]))

    def load_variable(self, state_dict, name, prefix=''):
        """加载单个变量的函数
        """
        variable = state_dict[name]
        if name in {'encoder.embed_tokens.weight', 'shared.weight'}:
            return self.load_embeddings(variable)
        else:
            return variable

    def variable_mapping(self, prefix=''):
        # 查看check_point发现'shared.weight'
        mapping = {f'{prefix}embeddings.word_embeddings.weight': 'encoder.embed_tokens.weight',
                   f'{prefix}encoderLayer.0.multiHeadAttention.relative_positions_encoding.weight': 'encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight',
                   f'{prefix}final_layer_norm.weight': 'encoder.final_layer_norm.weight'}
        for i in range(self.num_hidden_layers):
            mapping.update(
                {
                f'{prefix}encoderLayer.{i}.multiHeadAttention.q.weight': f'encoder.block.{i}.layer.0.SelfAttention.q.weight',
                f'{prefix}encoderLayer.{i}.multiHeadAttention.k.weight': f'encoder.block.{i}.layer.0.SelfAttention.k.weight',
                f'{prefix}encoderLayer.{i}.multiHeadAttention.v.weight': f'encoder.block.{i}.layer.0.SelfAttention.v.weight',
                f'{prefix}encoderLayer.{i}.multiHeadAttention.o.weight': f'encoder.block.{i}.layer.0.SelfAttention.o.weight',
                f'{prefix}encoderLayer.{i}.layerNorm1.weight': f'encoder.block.{i}.layer.0.layer_norm.weight',
                f'{prefix}encoderLayer.{i}.feedForward.outputDense.weight': f'encoder.block.{i}.layer.1.DenseReluDense.wo.weight',
                f'{prefix}encoderLayer.{i}.layerNorm2.weight': f'encoder.block.{i}.layer.1.layer_norm.weight',
                })

            if self.version.endswith('t5.1.0'):
                mapping.update({f'{prefix}encoderLayer.{i}.feedForward.intermediateDense.weight': f'encoder.block.{i}.layer.1.DenseReluDense.wi.weight'})
            elif self.version.endswith('t5.1.1'):
                mapping.update({f'{prefix}encoderLayer.{i}.feedForward.intermediateDense.weight': f'encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight',
                                f'{prefix}encoderLayer.{i}.feedForward.intermediateDense1.weight': f'encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight'})
        return mapping
    

class T5_Decoder(Decoder):
    @insert_arguments(version='t5.1.0')
    def __init__(self, *args, **kwargs):
        kwargs.update({'p_bias': 't5_relative', 'relative_attention_num_buckets': kwargs.get('relative_attention_num_buckets'), 'version': self.version,
                       'bias': False, 'norm_mode': 'rmsnorm'})  # p_bias来控制embedding阶段无pos_embedding，t5不使用bias，并且使用rmsnorm
        super().__init__(*args, **kwargs)
        del self.embeddings.layerNorm

        # t5的layernorm都在前面，因此重新定义了下
        layer = T5Layer(self.hidden_size, self.num_attention_heads, self.dropout_rate, self.attention_probs_dropout_prob, self.intermediate_size, self.hidden_act, is_dropout=self.is_dropout, 
                            conditional_size=self.conditional_size, is_decoder=True, **get_kw(BertLayer, kwargs))
        self.decoderLayer = nn.ModuleList([copy.deepcopy(layer) for _ in range(self.num_hidden_layers)])
        
        # 把第二层后的相对位置编码的权重绑定到第一层上，变相实现仅由第一层计算
        for i in range(1, self.num_hidden_layers):
            self.decoderLayer[i].multiHeadAttention.relative_positions_encoding.weight = self.decoderLayer[0].multiHeadAttention.relative_positions_encoding.weight
        self.final_layer_norm = LayerNorm(self.hidden_size, eps=1e-12, conditional_size=self.conditional_size, bias=False, norm_mode='rmsnorm')
        self.dropout = nn.Dropout(self.dropout_rate)

    def apply_final_layers(self, inputs):
        inputs[0][1] = self.dropout(self.final_layer_norm([inputs[0][1]]))  # 在转logit前把最后一层的hidden_states加layernorm
        return super().apply_final_layers(inputs)

    def load_variable(self, state_dict, name, prefix=''):
        """加载单个变量的函数
        """
        variable = state_dict[name]
        if name in {f'decoder.embed_tokens.weight', 'lm_head.weight', 'shared.weight'}:
            return self.load_embeddings(variable)
        else:
            return variable

    def variable_mapping(self, prefix=''):
        # 查看check_point发现'shared.weight'
        mapping = {f'{prefix}embeddings.word_embeddings.weight': 'decoder.embed_tokens.weight',
                   f'{prefix}decoderLayer.0.multiHeadAttention.relative_positions_encoding.weight': 'decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight',
                   f'{prefix}final_layer_norm.weight': 'decoder.final_layer_norm.weight',
                   f'{prefix}final_dense.weight': 'lm_head.weight'}

        for i in range(self.num_hidden_layers):
            mapping.update(
                {
                f'{prefix}decoderLayer.{i}.multiHeadAttention.q.weight': f'decoder.block.{i}.layer.0.SelfAttention.q.weight',
                f'{prefix}decoderLayer.{i}.multiHeadAttention.k.weight': f'decoder.block.{i}.layer.0.SelfAttention.k.weight',
                f'{prefix}decoderLayer.{i}.multiHeadAttention.v.weight': f'decoder.block.{i}.layer.0.SelfAttention.v.weight',
                f'{prefix}decoderLayer.{i}.multiHeadAttention.o.weight': f'decoder.block.{i}.layer.0.SelfAttention.o.weight',
                f'{prefix}decoderLayer.{i}.layerNorm1.weight': f'decoder.block.{i}.layer.0.layer_norm.weight',

                f'{prefix}decoderLayer.{i}.crossAttention.q.weight': f'decoder.block.{i}.layer.1.EncDecAttention.q.weight',
                f'{prefix}decoderLayer.{i}.crossAttention.k.weight': f'decoder.block.{i}.layer.1.EncDecAttention.k.weight',
                f'{prefix}decoderLayer.{i}.crossAttention.v.weight': f'decoder.block.{i}.layer.1.EncDecAttention.v.weight',
                f'{prefix}decoderLayer.{i}.crossAttention.o.weight': f'decoder.block.{i}.layer.1.EncDecAttention.o.weight',
                f'{prefix}decoderLayer.{i}.layerNorm3.weight': f'decoder.block.{i}.layer.1.layer_norm.weight',

                f'{prefix}decoderLayer.{i}.feedForward.outputDense.weight': f'decoder.block.{i}.layer.2.DenseReluDense.wo.weight',
                f'{prefix}decoderLayer.{i}.layerNorm2.weight': f'decoder.block.{i}.layer.2.layer_norm.weight',
                })

            if self.version.endswith('t5.1.0'):
                mapping.update({f'{prefix}decoderLayer.{i}.feedForward.intermediateDense.weight': f'decoder.block.{i}.layer.2.DenseReluDense.wi.weight'})
            elif self.version.endswith('t5.1.1'):
                mapping.update({f'{prefix}decoderLayer.{i}.feedForward.intermediateDense.weight': f'decoder.block.{i}.layer.2.DenseReluDense.wi_0.weight',
                                f'{prefix}decoderLayer.{i}.feedForward.intermediateDense1.weight': f'decoder.block.{i}.layer.2.DenseReluDense.wi_1.weight'})
        return mapping


class T5(Transformer):
    """Google的T5模型（Encoder-Decoder）
    """
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, *args,  tie_emb_src_tgt_weight=True, **kwargs):
        super(T5, self).__init__(*args, **kwargs)
        self.tie_emb_src_tgt_weight = tie_emb_src_tgt_weight

        # encoder
        self.encoder = T5_Encoder(*args, **kwargs)
        self.encoder.build(**kwargs)

        # decoder
        self.decoder = T5_Decoder(*args, **kwargs)
        self.decoder.build(**kwargs)

    def load_variable(self, state_dict, name, prefix=''):
        """加载单个变量的函数
        """
        variable = state_dict[name]
        if name in {'shared.weight', 'encoder.embed_tokens.weight', 'decoder.embed_tokens.weight', 'lm_head.weight'}:
            return self.load_embeddings(variable)
        else:
            return variable

    def variable_mapping(self, prefix=''):
        mapping = self.encoder.variable_mapping(prefix='encoder.')
        mapping.update(self.decoder.variable_mapping(prefix='decoder.'))
        if self.tie_emb_src_tgt_weight:
            mapping.update({'encoder.embeddings.word_embeddings.weight': 'shared.weight',
                            'decoder.embeddings.word_embeddings.weight': 'shared.weight'})
        return mapping


class GPT(LM_Mask, BERT):
    """构建GPT模型
    链接：https://github.com/openai/finetune-transformer-lm
    """
    @insert_arguments(final_activation='softmax')
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, max_position, **kwargs):
        """GPT的embedding是token、position、segment三者embedding之和，跟BERT的主要区别是三者相加之后没有加LayerNormalization层。
           使用LM_Mask实现预训练ckpt中的bias参数，最后的全连接层由于和embedding层权重一致，因此直接从word_embedding取
        """
        super(GPT, self).__init__(max_position, **kwargs)
        del self.embeddings.layerNorm
        self.dense = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.dense.weight = self.embeddings.word_embeddings.weight
        self.final_activation = get_activation(self.final_activation)

    def apply_final_layers(self, inputs):
        hidden_state = super().apply_final_layers(inputs)
        logit = self.dense(hidden_state)
        return self.final_activation(logit)

    def load_variable(self, state_dict, name):
        return super(GPT, self).load_variable(state_dict, name, prefix='gpt')

    def variable_mapping(self):
        """映射到GPT权重格式
        """
        mapping =  super(GPT, self).variable_mapping(prefix='gpt')
        return mapping


class GPT2(LM_Mask, BERT):
    """构建GPT模型
    链接：https://github.com/openai/finetune-transformer-lm
    """
    @insert_arguments(final_activation='softmax')
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, max_position, **kwargs):
        """GPT2的embedding是token、position两者embedding之和
           1、跟BERT的主要区别是三者相加之后没有加LayerNormalization层。
           2、bert的layernorm是在attn/ffc之后，OpenAi-gpt2是在之前。
           使用LM_Mask实现预训练ckpt中的bias参数，最后的全连接层由于和embedding层权重一致，因此直接从word_embedding取
        """
        super(GPT2, self).__init__(max_position, **kwargs)
        del self.embeddings.layerNorm
        layer = self.Gpt2Layer(self.hidden_size, self.num_attention_heads, self.dropout_rate, self.attention_probs_dropout_prob, self.intermediate_size, self.hidden_act, is_dropout=self.is_dropout, conditional_size=self.conditional_size)
        self.encoderLayer = nn.ModuleList([copy.deepcopy(layer) if layer_id in self.keep_hidden_layers else Identity() for layer_id in range(self.num_hidden_layers)])
        self.LayerNormFinal = LayerNorm(self.hidden_size, eps=1e-12, conditional_size=self.conditional_size)
        self.dense = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.dense.weight = self.embeddings.word_embeddings.weight
        self.final_activation = get_activation(self.final_activation)

    def apply_final_layers(self, inputs):
        hidden_state = super().apply_final_layers(inputs)
        logit = self.dense(self.LayerNormFinal([hidden_state]))
        return self.final_activation(logit)

    def load_variable(self, state_dict, name):
        return super(GPT2, self).load_variable(state_dict, name, prefix='gpt2')

    def variable_mapping(self):
        """映射到GPT权重格式
        """
        mapping =  super(GPT2, self).variable_mapping(prefix='gpt2')
        mapping.update({'LayerNormFinal.weight': 'gpt2.LayerNormFinal.weight',
                        'LayerNormFinal.bias': 'gpt2.LayerNormFinal.bias'})
        return mapping
    
    class Gpt2Layer(BertLayer):
        '''未定义在layer.py中是因为该层针对gpt2_mlm模型，不可复用
        顺序：LN --> Att --> Add --> LN --> FFN --> Add
        '''
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        def forward(self, hidden_states, attention_mask, conditional_emb=None, encoder_hidden_states=None, encoder_attention_mask=None):
            # bert的layernorm是在attn/ffc之后，Openai-gpt2是在之前
            x = self.layerNorm1((hidden_states, conditional_emb))
            self_attn_output = self.multiHeadAttention(x, attention_mask)
            hidden_states = hidden_states + self.dropout1(self_attn_output)
            x = self.layerNorm2((hidden_states, conditional_emb))
            ffn_output = self.feedForward(x)
            hidden_states = hidden_states + self.dropout2(ffn_output)
            return hidden_states


class GPT2_ML(LM_Mask, BERT):
    """构建GPT2_ML模型
    链接: https://github.com/imcaspar/gpt2-ml
    注意：GPT2_ML虽然号称GPT2，但是它的结构其实更接近GPT，它自称GPT2的原因大概是因为它开源的版本参数量达到了GPT2的15亿参数。
         看完ckpt中的key，和GPT的区别是embedding后也有layernorm，和bert的区别是第一个跳跃链接是在layernorm前，bert是在之后
    """
    @insert_arguments(final_activation='softmax')
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, max_position, **kwargs):
        super().__init__(max_position, **kwargs)
        layer = self.Gpt2MlLayer(self.hidden_size, self.num_attention_heads, self.dropout_rate, self.attention_probs_dropout_prob, self.intermediate_size, self.hidden_act, is_dropout=self.is_dropout, conditional_size=self.conditional_size)
        self.encoderLayer = nn.ModuleList([copy.deepcopy(layer) if layer_id in self.keep_hidden_layers else Identity() for layer_id in range(self.num_hidden_layers)])
        self.dense = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.dense.weight = self.embeddings.word_embeddings.weight
        self.final_activation = get_activation(self.final_activation)

    def apply_final_layers(self, inputs):
        hidden_state = super().apply_final_layers(inputs)
        logit = self.dense(hidden_state)
        return self.final_activation(logit)

    def load_variable(self, state_dict, name):
        return super(GPT2_ML, self).load_variable(state_dict, name, prefix='gpt2_ml')

    def variable_mapping(self):
        """映射到GPT2权重格式
        """
        mapping =  super(GPT2_ML, self).variable_mapping(prefix='gpt2_ml')
        return mapping

    class Gpt2MlLayer(BertLayer):
        '''未定义在layer.py中是因为该层针对gpt2_mlm模型，不可复用
        顺序：Att --> Add --> LN --> FFN --> Add --> LN
        '''
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        def forward(self, hidden_states, attention_mask, conditional_emb=None, encoder_hidden_states=None, encoder_attention_mask=None):
            self_attn_output = self.multiHeadAttention(hidden_states, attention_mask)
            hidden_states = hidden_states + self.dropout1(self_attn_output)
            x = self.layerNorm1((hidden_states, conditional_emb))
            # bert的跳跃连接是在layerNorm之后，gpt2_ml是在layerNorm之前
            ffn_output = self.feedForward(x)
            hidden_states = hidden_states + self.dropout2(ffn_output)
            hidden_states = self.layerNorm2((hidden_states, conditional_emb))
            return hidden_states


class Transformer_XL(BERT):
    '''构建transformer-xl模型, 已加载
    项目: https://github.com/kimiyoung/transformer-xl
    不同点:  
        1) 简化了原有的AdaptiveEmbedding(可选)和未使用ProjectedAdaptiveLogSoftmax, 直接输出last_hidden_state
        2) mems修改了transformer中初始化为zero_tensor, 改为包含最后一层, 原项目初始化为empty_tensor
        3) SinusoidalPositionEncoding一般是sincos间隔排列, 这里是先sin后cos
        4) attention_mask在multi_attn中使用中使用1e30来替代原来的1000
    '''
    @delete_arguments('with_pool', 'with_nsp', 'with_mlm')
    @insert_arguments(with_lm=False)
    def __init__(self, *args, mem_len=0, same_length=False, clamp_len=-1, **kwargs):
        # p_bias来控制embedding阶段无pos_embedding
        kwargs.update({'p_bias': 'other_relative'})
        super().__init__(*args, **kwargs)
        self.mem_len, self.same_length, self.clamp_len = mem_len, same_length, clamp_len
        self.attn_type = kwargs.get('attn_type', 0)

        # embedding
        if kwargs.get('adaptive_embedding'):
            cutoffs, div_val, sample_softmax = kwargs.get('cutoffs', []), kwargs.get('div_val', 1), kwargs.get('sample_softmax', False)
            self.embeddings = AdaptiveEmbedding(self.vocab_size, self.embedding_size, self.hidden_size, cutoffs, div_val, sample_softmax, **get_kw(AdaptiveEmbedding, kwargs))
        else:
            self.embeddings = nn.Embedding(self.vocab_size, self.embedding_size)
        self.pos_embeddings = XlnetPositionsEncoding(self.embedding_size)
        self.dropout = nn.Dropout(self.dropout_rate)

        # 每层自己的r_w_bias和r_r_bias，还是公用
        if not kwargs.get('untie_r'):
            self.r_w_bias = nn.Parameter(torch.FloatTensor(self.num_attention_heads, self.attention_head_size))  # 全局内容偏置
            self.r_r_bias = nn.Parameter(torch.FloatTensor(self.num_attention_heads, self.attention_head_size))  # 全局位置偏置
            if self.segment_vocab_size > 0:
                self.r_s_bias = nn.Parameter(torch.FloatTensor(self.num_attention_heads, self.attention_head_size))  # 全局segment偏置
        else:
            self.r_w_bias, self.r_r_bias = None, None
            self.r_s_bias = None

        # transformer block
        layer = XlnetLayer(self.hidden_size, self.num_attention_heads, self.dropout_rate, self.attention_probs_dropout_prob, self.intermediate_size, 
                           self.hidden_act, is_dropout=self.is_dropout, conditional_size=self.conditional_size, r_w_bias=self.r_w_bias, r_r_bias=self.r_r_bias,
                           r_s_bias=None, **get_kw(BertLayer, kwargs))
        self.encoderLayer = nn.ModuleList([copy.deepcopy(layer) if layer_id in self.keep_hidden_layers else Identity() for layer_id in range(self.num_hidden_layers)])

        # 映射
        if self.with_lm:
            self.dense = nn.Linear(self.hidden_size, self.vocab_size, bias=True)

    def init_mems(self, bsz):
        '''初始化mems, 用于记忆mlen的各层隐含层状态
        '''
        if isinstance(self.mem_len, (int, float)) and (self.mem_len > 0):
            mems = []
            param = next(self.parameters())
            for _ in range(self.num_hidden_layers+1):
                empty = torch.zeros(bsz, self.mem_len, self.hidden_size, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mlen, qlen):
        '''更新mems
        '''
        # does not deal with None
        if self.mems is None:
            return None
        # mems is not None
        assert len(hids) == len(self.mems), "len(hids) != len(mems)"
        # There are `mlen + qlen` steps that can be cached into mems
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):
                cat = torch.cat([self.mems[i], hids[i]], dim=1)
                new_mems.append(cat[:, beg_idx:end_idx].detach())
        self.mems = new_mems

    def relative_positional_encoding(self, qlen, klen, device):
        # 生成pos_emb, 这里使用sincos的位置编码，为了和xlnet入参一致
        pos_seq = torch.arange(klen-1, -1, -1.0, device=device, dtype=torch.long)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        pos_emb = self.dropout(self.pos_embeddings(pos_seq))  # 用word_emb的dropout
        return pos_emb

    def create_mask(self, word_emb, qlen, klen, mlen):
        # 修改attention_mask, mlen可以全部访问，q_len只能访问<=t时刻的, mask和Unilm类似，但是Unilm是靠segement_ids来控制
        if self.same_length:  # 只能访问前面固定长度
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            mask_shift_len = qlen - mask_len if mask_len > 0 else qlen
            attention_mask = 1-(torch.triu(all_ones, 1+mlen) + torch.tril(all_ones, -mask_shift_len)).byte() # -1
        else:
            attention_mask = torch.tril(word_emb.new_ones(qlen, klen), diagonal=mlen).byte()  # [q_len, k_len], 下三角为1矩阵
        attention_mask = attention_mask[None, None, :, :]
        return attention_mask

    def apply_embeddings(self, inputs):
        '''接受的inputs输入: [token_ids, segment_ids], 暂不支持条件LayerNorm输入
        '''
        assert isinstance(inputs, (list, tuple)), f'Inputs only support list,tuple format but passed {type(inputs)}'

        self.mems = self.init_mems(inputs[0].size(0))  # 生成mems
        # 精简后embeddings中只计算word_emdedding
        word_emb = self.dropout(self.embeddings(inputs[0]))
        index_ = 1
        btz, qlen = inputs[0].shape[:2]  # query长度
        mlen = self.mems[0].size(1) if self.mems is not None else 0
        klen = mlen + qlen
        # 相对位置编码
        pos_emb = self.relative_positional_encoding(qlen, klen, word_emb.device)
        # segment embedding
        if self.segment_vocab_size > 0:
            segment_ids = inputs[index_]
            if mlen > 0:
                mem_pad = torch.zeros([btz, mlen], dtype=torch.long, device=word_emb.device)
                cat_ids = torch.cat([mem_pad, segment_ids], dim=1)
            else:
                cat_ids = segment_ids
            # `1` indicates not in the same segment [qlen x klen x bsz]
            segment_ids = (segment_ids[:, :, None] != cat_ids[:, None]).long()
            index_ += 1
        else:
            segment_ids = None

        if self.attn_type in {'uni', 0}:  # 兼容transformer_xl的设置: 0
            attention_mask = self.create_mask(word_emb, qlen, klen, mlen)
        elif self.attn_type == 'bi':
            attention_mask = (inputs[0] != self.token_pad_ids).long().unsqueeze(1).unsqueeze(2)
        non_tgt_mask = torch.eye(qlen).to(attention_mask)[None, None, :, :]
        non_tgt_mask = ((1 - attention_mask - non_tgt_mask) <= 0).long()

        return [word_emb, segment_ids, pos_emb, non_tgt_mask, None]

    def apply_main_layers(self, inputs):
        hidden_states, segment_ids, pos_emb, attention_mask, conditional_emb = inputs[:5]
        encoded_layers = [hidden_states] # 添加embedding的输出

        layer_inputs = [hidden_states, segment_ids, pos_emb, attention_mask, None, conditional_emb]
        for i, layer_module in enumerate(self.encoderLayer):
            mems_i = None if self.mems is None else self.mems[i]
            layer_inputs[-2] = mems_i
            layer_inputs = self.apply_on_layer_begin(i, layer_inputs)
            hidden_states = layer_module(*layer_inputs)
            layer_inputs[0] = hidden_states
            layer_inputs = self.apply_on_layer_end(i, layer_inputs)
            encoded_layers.append(hidden_states)
        
        # 原实现中word_emb, pos_emb和core_out(hidden_states)使用同一个dropout
        hidden_states = self.dropout(hidden_states)
        qlen = inputs[0].size(1)  # query长度
        mlen = self.mems[0].size(0) if self.mems is not None else 0
        self._update_mems(encoded_layers, mlen, qlen)
        
        if not self.output_all_encoded_layers:
            # 不返回所有层，即返回顶层
            encoded_layers = encoded_layers[:1] + [hidden_states]
        return [encoded_layers, conditional_emb]
    
    def load_variable(self, state_dict, name, prefix=''):
        # 这里由于预训练模型使用了AdapterEmbedding，因此暂不支持
        if (self.keep_tokens is not None) or (self.compound_tokens is not None):
            raise ValueError('Custom keep_tokens and compound_tokens is not yet supported in Transformer_XL')
        return state_dict[name]

    def variable_mapping(self, prefix=''):
        return {k:k for k, v in self.named_parameters()}

class XLNET(Transformer_XL):
    '''构建xlnet模型, 这里做了简化, 只用来finetune, 即没有perm_mask, target_mapping这些输入
       接受的inputs输入: [token_ids, segment_ids]
    '''
    def __init__(self, *args, bi_data=False, **kwargs):
        self.attn_type = kwargs.get('attn_type', 'bi')
        self.bi_data = bi_data
        kwargs['rel_shift_opt'] = 'xlnet'
        super().__init__(*args, **kwargs)
    
    def relative_positional_encoding(self, qlen, klen, device):
        # 生成pos_emb, 这里使用sincos的位置编码, transformer_xl里面有-1
        if self.attn_type == 'bi':
            beg, end = klen, -qlen
        elif self.attn_type == "uni":
            beg, end = klen, -1
        else:
            raise ValueError(f"Unknown `attn_type` {self.attn_type}.") 

        # 前向的emb
        pos_seq = torch.arange(beg, end, -1.0, device=device, dtype=torch.long)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        fwd_pos_emb = self.pos_embeddings(pos_seq)

        # 双向数据
        if self.bi_data:
            pos_seq = torch.arange(-beg, -end, -1.0, device=device, dtype=torch.long)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            bwd_pos_emb = self.pos_embeddings(pos_seq)
            pos_emb = torch.cat([fwd_pos_emb, bwd_pos_emb], dim=0)
        else:
            pos_emb = fwd_pos_emb

        pos_emb = self.dropout(pos_emb)  # 用word_emb的dropout
        return pos_emb

    def apply_final_layers(self, inputs):
        hidden_state = super().apply_final_layers(inputs)
        if self.with_lm:
            return [hidden_state, self.dense(hidden_state)]
        else:
            return hidden_state

    def load_variable(self, state_dict, name, prefix='transformer'):
        """加载单个变量的函数
        """
        variable = state_dict[name]
        if name in {f'{prefix}.word_embedding.weight', 'lm_loss.weight', 'lm_loss.bias'}:
            return self.load_embeddings(variable)
        elif re.search('rel_attn\.(q|k|v|r)$', name):
            return variable.reshape(variable.shape[0], -1).T
        # elif re.search('rel_attn\.(o|seg_embed)$', name):
        elif re.search('rel_attn\.(o)$', name):
            return variable.reshape(variable.shape[0], -1)
        else:
            return variable

    def variable_mapping(self, prefix='transformer'):
        mapping = {
            'embeddings.weight': f'{prefix}.word_embedding.weight',
            'dense.weight': 'lm_loss.weight',
            'dense.bias': 'lm_loss.bias',
        }
        for i in range(self.num_hidden_layers):
            prefix_i = f'{prefix}.layer.%d.' % i
            mapping.update({f'encoderLayer.{i}.multiHeadAttention.q.weight': prefix_i + 'rel_attn.q',
                            f'encoderLayer.{i}.multiHeadAttention.k.weight': prefix_i + 'rel_attn.k',
                            f'encoderLayer.{i}.multiHeadAttention.v.weight': prefix_i + 'rel_attn.v',
                            f'encoderLayer.{i}.multiHeadAttention.o.weight': prefix_i + 'rel_attn.o',
                            f'encoderLayer.{i}.multiHeadAttention.r.weight': prefix_i + 'rel_attn.r',
                            f'encoderLayer.{i}.multiHeadAttention.r_r_bias': prefix_i + 'rel_attn.r_r_bias',
                            f'encoderLayer.{i}.multiHeadAttention.r_s_bias': prefix_i + 'rel_attn.r_s_bias',
                            f'encoderLayer.{i}.multiHeadAttention.r_w_bias': prefix_i + 'rel_attn.r_w_bias',
                            # f'encoderLayer.{i}.multiHeadAttention.seg_embed.weight': prefix_i + 'rel_attn.seg_embed',
                            f'encoderLayer.{i}.multiHeadAttention.seg_embed': prefix_i + 'rel_attn.seg_embed',
                            f'encoderLayer.{i}.layerNorm1.weight': prefix_i + 'rel_attn.layer_norm.weight',
                            f'encoderLayer.{i}.layerNorm1.bias': prefix_i + 'rel_attn.layer_norm.bias',
                            f'encoderLayer.{i}.feedForward.intermediateDense.weight': prefix_i + 'ff.layer_1.weight',
                            f'encoderLayer.{i}.feedForward.intermediateDense.bias': prefix_i + 'ff.layer_1.bias',
                            f'encoderLayer.{i}.feedForward.outputDense.weight': prefix_i + 'ff.layer_2.weight',
                            f'encoderLayer.{i}.feedForward.outputDense.bias': prefix_i + 'ff.layer_2.bias',
                            f'encoderLayer.{i}.layerNorm2.weight': prefix_i + 'ff.layer_norm.weight',
                            f'encoderLayer.{i}.layerNorm2.bias': prefix_i + 'ff.layer_norm.bias'
                            })

        return mapping


def build_transformer_model(
        config_path=None,
        checkpoint_path=None,
        model='bert',
        application='encoder',
        **kwargs
):
    """根据配置文件构建模型，可选加载checkpoint权重
    """
    configs = {}
    if config_path is not None:
        configs.update(json.load(open(config_path)))
    configs.update(kwargs)
    if 'max_position' not in configs:
        configs['max_position'] = configs.get('max_position_embeddings', 512)
    if 'dropout_rate' not in configs:
        configs['dropout_rate'] = configs.get('hidden_dropout_prob')
    if 'segment_vocab_size' not in configs:
        configs['segment_vocab_size'] = configs.get('type_vocab_size', 2)
    
    models = {
        'bert': BERT,
        'roberta': BERT,  
        'albert': ALBERT,
        'albert_unshared': ALBERT_Unshared,
        'nezha': NEZHA,
        'roformer': RoFormer,
        'roformer_v2': RoFormerV2,
        'gau_alpha': GAU_alpha,
        'electra': ELECTRA,
        'ernie': ERNIE,
        'encoder': Encoder,
        'decoder': Decoder,
        'transformer': Transformer,
        'bart': BART,
        'gpt': GPT,
        'gpt2': GPT2,
        'gpt2_ml': GPT2_ML,
        't5': T5,
        't5_encoder': T5_Encoder,
        't5_decoder': T5_Decoder,
        't5.1.0': T5,
        't5.1.0_encoder': T5_Encoder,
        't5.1.0_decoder': T5_Decoder,
        't5.1.1': T5,
        't5.1.1_encoder': T5_Encoder,
        't5.1.1_decoder': T5_Decoder,
        'mt5.1.1': T5,
        'mt5.1.1_encoder': T5_Encoder,
        'mt5.1.1_decoder': T5_Decoder,
        'transformer_xl': Transformer_XL,
        'xlnet': XLNET,
    }

    if isinstance(model, str):  # string表示使用自带的模型
        MODEL = models[model.lower()]
        if model.endswith('t5.1.1'):
            configs['version'] = model
    elif isinstance(model, type) and issubclass(model, BERT_BASE): # nn.Module表示使用自定义的模型：
        MODEL = model
    else:
        raise ValueError('"model" args type should be string or nn.Module')

    application = application.lower()
    if application in ['lm', 'unilm'] and model in ['electra', 't5', ]:
        raise ValueError(f'"{model}" model can not be used as "{application}" application.\n')

    if application == 'lm':
        MODEL = extend_with_language_model(MODEL)
    elif application == 'unilm':
        MODEL = extend_with_unified_language_model(MODEL)

    transformer = MODEL(**configs)
    transformer.build(**configs)
    transformer.apply(transformer.init_model_weights)  # 初始化权重

    if checkpoint_path is not None:
        transformer.load_weights_from_pytorch_checkpoint(checkpoint_path)   
    transformer.configs = configs
    return transformer