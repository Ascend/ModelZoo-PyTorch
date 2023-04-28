# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

AISHELL-1 transformer model recipe. (Adapted from the LibriSpeech recipe.)

"""
import json
import sys
import logging
import time
import ssl
import torch
import numpy as np
import onnxruntime as rt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
from ais_bench.infer.interface import InferSession
import speechbrain as sb

ssl._create_default_https_context = ssl._create_unverified_context

logger = logging.getLogger(__name__)


# Define training procedure
class ASR(sb.core.Brain):
    def __init__(
        self,
        modules=None,
        opt_class=None,
        hparams=None,
        run_opts=None,
        checkpointer=None,
        profiler=None,
    ):
        super().__init__(
            modules=modules,
            opt_class=opt_class,
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=checkpointer,
            profiler=profiler,
        )


    def export_encoder_onnx(self, src, tokens_bos, wav_lens, pad_index):
        pad_index = torch.from_numpy(np.array([pad_index])).to(self.device)
        torch.onnx.export(
            self.hparams.Transformer,
            (src, tokens_bos, wav_lens, pad_index),
            'encoder.onnx',
            input_names=['src', 'tokens_bos', 'wav_lens', 'pad_index'],
            output_names=['enc_out'],
            dynamic_axes={
                "src": {0: "batch_size", 1: "wav_len"},
                "tokens_bos": {0: "batch_size", 1: "token_len"},
                "wav_lens": {0: "batch_size"}},
            do_constant_folding=True,
            opset_version=11
        )

    def export_decoder_onnx(self, tgt,encoder_out, tgt_mask, src_key_padding_mask, pos_embs_target, pos_embs_encoder):
        torch.onnx.export(
            self.hparams.Transformer.decoder,
            (tgt, encoder_out, tgt_mask, None,None, src_key_padding_mask, pos_embs_target, pos_embs_encoder, False),
            'decoder.onnx',
            input_names=['token', 'encoder_out', 'decoder_mask', 'None1', 'None2',
             'encoder_mask', 'pos_embs_target', 'pos_embs_encoder'],
            output_names=['output'],
            dynamic_axes={"token": {0: "batch_size", 1: "seq_len"},
                        "encoder_out": {0: "batch_size", 1: "wav_len"},
                        "decoder_mask": {0: "batch_size", 1: "seq_len"}},
            do_constant_folding=True,
            opset_version=11
        )
    #src, tokens_bos, wav_lens, self.hparams.pad_index, encoder_model, decoder_model
    def encoder_om_infer(self, src, tokens_bos, wav_lens, pad_index, encoder_model):

        src_np = src.cpu().numpy()

        src_np = np.ascontiguousarray(src_np)

        tokens_bos_np = tokens_bos.cpu().numpy()

        wav_lens_np = wav_lens.cpu().numpy()

        pad_index_np = np.expand_dims(np.array(pad_index), axis=0)

        outputSizes = 10000000

        result = encoder_model.infer([src_np, wav_lens_np], 'dymshape', custom_sizes=outputSizes)

        enc_out_om = torch.from_numpy(result[0]).to(self.device)

        return enc_out_om
#batch, stage, device_id, encoder_model, decoder_model
    def compute_forward(self, batch, stage, device_id,encoder_model, decoder_model):
        """Forward computations from the waveform batches to the output probabilities."""

        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
                tokens_bos = torch.cat([tokens_bos, tokens_bos], dim=0)

        # compute features


        feats = self.hparams.compute_features(wavs)

        current_epoch = self.hparams.epoch_counter.current

        feats = self.hparams.normalize(feats, wav_lens, epoch=current_epoch)


        # forward modules
        src = self.hparams.CNN(feats)

        # Transformer om infer
        enc_out = self.encoder_om_infer(src, tokens_bos, wav_lens, self.hparams.pad_index, encoder_model)

        # Compute outputs
        hyps = None
        if stage == sb.Stage.TRAIN:
            hyps = None
        elif stage == sb.Stage.VALID:
            hyps = None
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch % self.hparams.valid_search_interval == 0:
                # for the sake of efficeincy, we only perform beamsearch with limited capacity
                # and no LM to give user some idea of how the AM is doing
                hyps, _ = self.hparams.valid_search(enc_out.detach(), wav_lens)
        elif stage == sb.Stage.TEST:

            hyps, _ = self.hparams.test_search(decoder_model, enc_out.detach(), wav_lens, run_opts['ctc_enable'])

        return None, None, None, hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        (
            p_ctc,
            p_seq,
            wav_lens,
            hyps,
        ) = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        if stage != sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval

            if current_epoch % valid_search_interval == 0 or (stage == sb.Stage.TEST):
                # Decode token terms to words
                predicted_words = [
                    tokenizer.decode_ids(utt_seq).split(" ") for utt_seq in hyps
                ]
                target_words = [wrd.split(" ") for wrd in batch.wrd]
                if self.hparams.remove_spaces:
                    predicted_words = ["".join(p) for p in predicted_words]
                    target_words = ["".join(t) for t in target_words]
                    self.cer_metric.append(ids, predicted_words, target_words)
            # compute the accuracy of the one-step-forward prediction
        return None

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        # check if we need to switch optimizer
        # if so change the optimizer from Adam to SGD
        self.check_and_reset_optimizer()

        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)

        # normalize the loss by gradient_accumulation step
        (loss / self.hparams.gradient_accumulation).backward()

        if self.step % self.hparams.gradient_accumulation == 0:
            # gradient clipping & early stop if loss is not fini
            self.check_gradients(loss)

            self.optimizer.step()
            self.optimizer.zero_grad()

            # anneal lr every update
            self.hparams.noam_annealing(self.optimizer)

        return loss.detach()
#batch, Stage.TEST, device_id, encoder_model, decoder_model, out_que
    def evaluate_batch(self, batch, stage, device_id, encoder_model, decoder_model, out_que):
        """Computations needed for validation/test batches"""
        with torch.no_grad():

            predictions = self.compute_forward(batch, stage, device_id, encoder_model, decoder_model)
            loss = self.compute_objectives(predictions, batch, stage=stage)
            out_que.put(loss)
        return out_que

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.cer_metric = self.hparams.cer_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0 or stage == sb.Stage.TEST:
                stage_stats["CER"] = self.cer_metric.summarize("error_rate")

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():

            # report different epoch stages acccording current stage
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch <= self.hparams.stage_one_epochs:
                lr = self.hparams.noam_annealing.current_lr
                steps = self.hparams.noam_annealing.n_steps
                optimizer = self.optimizer.__class__.__name__
            else:
                lr = self.hparams.lr_sgd
                steps = -1
                optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "steps": steps,
                "optimizer": optimizer,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ACC": stage_stats["ACC"], "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=10,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.cer_file, "w") as w:
                self.cer_metric.write_stats(w)

            # save the averaged checkpoint at the end of the evalation stage
            # delete the rest of the intermediate checkpoints
            # ACC is set to 1.1 so checkpointer only keeps the averaged checkpoint
            self.checkpointer.save_and_keep_only(
                meta={"ACC": 1.1, "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=1,
            )

    def check_and_reset_optimizer(self):
        """reset the optimizer if training enters stage 2"""
        current_epoch = self.hparams.epoch_counter.current
        if not hasattr(self, "switched"):
            self.switched = False
            if isinstance(self.optimizer, torch.optim.SGD):
                self.switched = True

        if self.switched is True:
            return

        if current_epoch > self.hparams.stage_one_epochs:
            self.optimizer = self.hparams.SGD(self.modules.parameters())

            if self.checkpointer is not None:
                self.checkpointer.add_recoverable("optimizer", self.optimizer)

            self.switched = True

    def on_fit_start(self):
        """Initilaize the right optimizer on the training start"""
        super().on_fit_start()

        # if the model is resumed from stage two, reinitilaize the optimizer
        current_epoch = self.hparams.epoch_counter.current
        current_optimizer = self.optimizer
        if current_epoch > self.hparams.stage_one_epochs:
            del self.optimizer
            self.optimizer = self.hparams.SGD(self.modules.parameters())

            # Load latest checkpoint to resume training if interrupted
            if self.checkpointer is not None:

                # do not reload the weights if training is interrupted right before stage 2
                group = current_optimizer.param_groups[0]
                if "momentum" not in group:
                    return

                self.checkpointer.recover_if_possible(device=torch.device(self.device))

    def on_evaluate_start(self, max_key=None, min_key=None):
        """perform checkpoint averge if needed"""
        super().on_evaluate_start()

        ckpts = self.checkpointer.find_checkpoints(max_key=max_key, min_key=min_key)
        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="model", device=self.device
        )

        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_data"],
        replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(sort_key="duration", reverse=True)
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError("sorting must be random, ascending or descending")

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_data"],
        replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_data"],
        replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]

    # Defining tokenizer and loading it
    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("transcript")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"],
    )
    return train_data, valid_data, test_data, tokenizer


if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # 1.  # Dataset prep (parsing Librispeech)
    from prepare import prepare_aishell  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_aishell,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["output_folder"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_data, tokenizer = dataio_prepare(hparams)

    # We download and pretrain the tokenizer
    run_on_main(hparams["pretrainer"].collect_files)
    run_opts["device"] = 'cpu'
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    test_duration = 0
    for i in test_data:
        test_duration += test_data.data[i['id']]['duration']
    print(f'test audio duration: {test_duration} s')

    if run_opts['mode'] == 'export':
        asr_brain.on_evaluate_start()
        print('encoder export')
        src = torch.rand((8,55,20,256), dtype=torch.float32)
        tokens_bos = torch.randint(0, 100, (8,9), dtype=torch.int64)
        wav_lens = torch.tensor((0.8568, 0.8995, 0.9120, 0.9448, 0.9613, 0.9844, 0.9935, 1.0000), dtype=torch.float32)
        pad_index = torch.tensor(0)
        asr_brain.export_encoder_onnx(src, tokens_bos, wav_lens, pad_index)

        print('decoder export')
        tgt = torch.rand((80, 1, 256), dtype=torch.float32)
        encoder_out = torch.rand((80, 55, 256), dtype=torch.float32)
        tgt_mask = torch.rand((1, 1), dtype=torch.float32)
        asr_brain.export_decoder_onnx(tgt, encoder_out, tgt_mask, None, None, None)
        
    elif run_opts['mode'] == 'infer':
        # adding objects to trainer:
        asr_brain.tokenizer = tokenizer

        # Testing
        infer_start = time.time()
        asr_brain.evaluate(test_data, test_loader_kwargs=hparams["test_dataloader_opts"], encoder_path=run_opts["encoder_file"], decopder_path=run_opts["decoder_file"])
        infer_end = time.time()
        infer_time = infer_end - infer_start
        RTF = (infer_end - infer_start) / test_duration
        print(f'infer time: {infer_time} s')
        print(f'RTF: {RTF}')
        performance_data = dict()
        performance_data['infer_time']  = infer_time
        performance_data['RTF']  = RTF
        filename='performance_of_{}.json'.format(run_opts['npu_rank'])
        with open(filename,'w') as file_obj:
            json.dump(performance_data,file_obj)
            file_obj.write('\n')