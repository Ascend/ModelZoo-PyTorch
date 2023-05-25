# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List
import numpy as np
import os
from random import choice

from torch import nn
import torch
if torch.__version__ >= '1.8':
    import torch_npu
import torch.nn.functional as F

import linklink as link
from prototype.model.clip import CLIP
from prototype.model.declip import projection_MLP, prediction_MLP
from prototype.model.utils.nnclr_modules.memory_bank import MemoryBankModule

class NNMemoryBankModule(MemoryBankModule):
    def __init__(self, size: int = 2 ** 16, topk: int = 1):
        super(NNMemoryBankModule, self).__init__(size)
        self.topk = topk

    def forward(self,
                output: torch.Tensor,
                update: bool = False):
        output, bank = super(NNMemoryBankModule, self).forward(output, update=update)
        bank = bank.to(output.device).t()

        output_normed = torch.nn.functional.normalize(output, dim=1)
        bank_normed = torch.nn.functional.normalize(bank, dim=1)

        similarity_matrix = output_normed @ bank_normed.t()

        _, index_nearest_neighbours = torch.topk(similarity_matrix, k=self.topk, dim=1)
        nearest_neighbours = [torch.index_select(bank, dim=0, index=index_nearest_neighbours[:,i]) for i in range(self.topk)]

        return nearest_neighbours

class DECLIP(CLIP):
    def __init__(self,image_encode, text_encode, use_allgather, nn_size=2**16, nn_topk=1, \
                 return_dense=False, return_simsiam_text=False, return_simsiam_nn_text=False, return_caption=False, return_nn_bank=False, text_mask_type=None,
                 EDA=True, feature_dim=1024, forward_type='split'):
        super(DECLIP, self).__init__(image_encode, text_encode, use_allgather)
        # TODO change for r50 checkpoint
        self.projector = projection_MLP(feature_dim)
        # self.projector = projection_MLP(1024)
        self.predictor = prediction_MLP(1024)
        self.return_dense = return_dense
        self.return_simsiam_nn_text = return_simsiam_nn_text
        self.return_nn_bank = return_nn_bank
        self.return_caption = return_caption
        self.return_simsiam_text = return_simsiam_text
        self.return_simsiam_nn_text = return_simsiam_nn_text
        self.text_mask_type = text_mask_type
        self.EDA = EDA
        self.forward_type = forward_type

        # follow official repo code
        from textaugment import EDA
        self.emd = EDA()

        if self.return_dense:
            raise NotImplementedError('These are bugs in the model, Please Check The Codes!')
            self.projector_d = projection_MLP(2048)  # dense
            self.predictor_d = prediction_MLP(1024)
        if self.return_simsiam_text:
            self.projector_text = projection_MLP(feature_dim)
            self.predictor_text = prediction_MLP(1024)
        if self.return_simsiam_nn_text:
            self.projector_nn_text = projection_MLP(feature_dim)
            self.predictor_nn_text = prediction_MLP(1024)
        if self.return_caption:
            raise NotImplementedError('Not Available')
        if text_mask_type is not None:
            enc_dim = self.encode_text.text_projection.weight.shape[-1]
            self.text_label_predictor = nn.Linear(enc_dim, self.encode_text.vocab_size)
        if self.return_nn_bank:
            self.nn_replacer_img = NNMemoryBankModule(size=nn_size, topk=nn_topk)
            self.nn_replacer_text = NNMemoryBankModule(size=nn_size, topk=nn_topk)

    def text_modules(self):
        ret = super(self).text_modules()
        if self.text_mask_type is not None:
           ret.append(self.text_label_predictor)
        return ret

    def visual_modules(self):
        ret = super(self).visual_modules()
        ret.extend([self.predictor, self.projector])
        return ret

    def encode_image(self, image, return_dense=False):
        if return_dense:
            output = self.visual(image.type(self.dtype), return_dense=return_dense)
            return output
        output = self.visual(image.type(self.dtype))
        return output


    def forward(self, input, return_dict=False):
        # input
        images = input['images']
        images_1, images_2 = torch.split(images, [3,3], dim=1)
        texts = input['captions']
        #clip encode
        texts = self.sample_captions(texts)
        texts_aug = []
        for caption in texts:
            if self.EDA:
                emd_aug = choice([self.emd.synonym_replacement, self.emd.random_swap, self.emd.random_deletion])
                cap_new = emd_aug(caption)
                if isinstance(cap_new, list):
                    cap_new = ' '.join(cap_new)
                texts_aug.append(cap_new)   # single word: there is a bug
            else:
                raise NotImplementedError('No EDA')

        if self.text_mask_type is not None:
            text_features, word_features, text_labels = self.encode_text(texts, mask_type = self.text_mask_type)
            text_features_aug = self.encode_text(texts_aug)
        else:
            text_features = self.encode_text(texts)
            if self.EDA:
                text_features_aug = self.encode_text(texts_aug)
            else:
                text_features_aug = text_features.detach()

        if not self.return_dense:
            if self.forward_type == 'image_concat':
                image_concat = torch.cat([images_1, images_2], dim=0)
                image_features_concat = self.encode_image(image_concat)
                image_features_1, image_features_2 = torch.split(image_features_concat, images_1.shape[0], dim=0)
            else:
                image_features_1 = self.encode_image(images_1)
                image_features_2 = self.encode_image(images_2)
        else:
            image_features_1, image_features_d1 = self.encode_image(images_1, return_dense=return_dense)
            image_features_2, image_features_d2 = self.encode_image(images_2, return_dense=return_dense)

        #simsiam
        z1 = self.projector(image_features_1)
        z2 = self.projector(image_features_2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        if self.return_dense:
            b, n = image_features_d1.shape[:2]
            #simsiam_dense
            z1d = self.projector_d(image_features_d1.reshape(b*n, -1))
            z2d = self.projector_d(image_features_d2.reshape(b*n, -1))
            p1d = self.predictor_d(z1d).reshape(b,n,-1)
            p2d = self.predictor_d(z2d).reshape(b,n,-1)
            z1d, z2d = z1d.reshape(b,n,-1), z2d.reshape(b,n,-1)


        # normalized features
        image_features_1 = image_features_1 / (image_features_1.norm(dim=-1, keepdim=True))
        image_features_2 = image_features_2 / (image_features_2.norm(dim=-1, keepdim=True))
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True)+1e-10)
        text_features_aug = text_features_aug / (text_features_aug.norm(dim=-1, keepdim=True)+1e-10)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logit_scale.data = torch.clamp(logit_scale.data, max=100)

        ## TODO there is a bug when nz format tensro do all_gather, avoiding whith nd format..
        text_features = torch_npu.npu_format_cast(text_features, 2)
        text_features_aug = torch_npu.npu_format_cast(text_features_aug, 2)

        if self.training and self.use_allgather:
            link.barrier()
            gathered_image_features_1 = self.all_gather(image_features_1)
            gathered_image_features_2 = self.all_gather(image_features_2)
            gathered_text_features = self.all_gather(text_features)
            gathered_text_features_aug = self.all_gather(text_features_aug)
            link.barrier()

            logits_per_image_1 = logit_scale * image_features_1 @ gathered_text_features.t()
            logits_per_image_2 = logit_scale * image_features_2 @ gathered_text_features.t()
            logits_per_image_1_aug = logit_scale * image_features_1 @ gathered_text_features_aug.t()
            logits_per_image_2_aug = logit_scale * image_features_2 @ gathered_text_features_aug.t()

            logits_per_text_1 = logit_scale * text_features @ gathered_image_features_1.t()
            logits_per_text_2 = logit_scale * text_features @ gathered_image_features_2.t()
            logits_per_text_1_aug = logit_scale * text_features_aug @ gathered_image_features_1.t()
            logits_per_text_2_aug = logit_scale * text_features_aug @ gathered_image_features_2.t()

            if self.return_nn_bank:
                text_features_nn = self.nn_replacer_text(text_features.detach().float(), update=False)
                text_features_nn = [t_feat.type(self.dtype) for t_feat in text_features_nn]
                text_features_nn = [t_feat / (t_feat.norm(dim=-1, keepdim=True) + 1e-10) for t_feat in text_features_nn]
                text_features_nn_aug = self.nn_replacer_text(text_features_aug.detach().float(), update=True)
                text_features_nn_aug = [t_feat.type(self.dtype) for t_feat in text_features_nn_aug]
                text_features_nn_aug = [t_feat / (t_feat.norm(dim=-1, keepdim=True) + 1e-10) for t_feat in text_features_nn_aug]
                self.nn_replacer_text(text_features.detach().float(), update=True)  # update

                gathered_text_features_nn = [self.all_gather(t_feat) for t_feat in text_features_nn]
                gathered_text_features_nn_aug = [self.all_gather(t_feat) for t_feat in text_features_nn_aug]

                logits_per_image_1_nn = torch.cat([logit_scale * image_features_1 @ t_feat.t()
                                                  for t_feat in gathered_text_features_nn])
                logits_per_image_2_nn = torch.cat([logit_scale * image_features_2 @ t_feat.t()
                                                  for t_feat in gathered_text_features_nn])
                logits_per_image_1_nn_aug = torch.cat([logit_scale * image_features_1 @ t_feat.t()
                                                      for t_feat in gathered_text_features_nn_aug])
                logits_per_image_2_nn_aug = torch.cat([logit_scale * image_features_2 @ t_feat.t()
                                                      for t_feat in gathered_text_features_nn_aug])
        else:
            raise NotImplementedError('2-View: Not Implemented')

        if return_dict:
            link.barrier()
            ret_dict = {}
            ret_dict['logits'] = logits_per_image_1, logits_per_image_2, logits_per_text_1, logits_per_text_2
            ret_dict['logits_aug'] = logits_per_image_1_aug, logits_per_image_2_aug, logits_per_text_1_aug, logits_per_text_2_aug
            ret_dict['simsiam_features'] = p1, p2, z1, z2
            ret_dict['features'] = text_features, image_features_1, image_features_2

            if self.return_simsiam_nn_text:
                #simsiam_text
                z_text = self.projector_nn_text(text_features)
                z_text_nn = [self.projector_nn_text(t_feat) for t_feat in text_features_nn]
                p_text = self.predictor_nn_text(z_text)
                ret_dict['nn_text_simsiam'] = p_text, z_text_nn
            if self.return_simsiam_text:
                z1t = self.projector(text_features)
                z2t = self.projector(text_features_aug)
                p1t = self.predictor(z1t)
                p2t = self.predictor(z2t)
                ret_dict['text_simsiam'] = p1t, p2t, z1t, z2t
            if self.return_nn_bank:
                ret_dict['nn_text_logits'] = logits_per_image_1_nn, logits_per_image_2_nn, logits_per_image_1_nn_aug, logits_per_image_2_nn_aug
            if self.text_mask_type is not None:
                if self.encode_text.text_encode_type == 'Bert':
                    text_pred_mask = word_features
                else:  # 30000
                    text_pred_mask = self.text_label_predictor(word_features)

                pred_mask = (text_labels != -100)
                mlm = F.cross_entropy(text_pred_mask[pred_mask], text_labels[pred_mask].to(text_pred_mask.device), reduce=None)
                ret_dict['text_self_supervised'] = mlm.mean()
            return ret_dict
        raise NotImplementedError('Must Return A Dict')
