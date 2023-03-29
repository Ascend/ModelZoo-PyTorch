# Copyright 2022 Huawei Technologies Co., Ltd
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
# ============================================================================
import argparse
import os
import sys
import torch
import numpy as np

sys.path.append('./ACE/')
from flair.utils.from_params import Params
from flair.config_parser import ConfigParser, log
from flair.trainers import ReinforcementTrainer
from flair.custom_data_loader import ColumnDataLoader
test_number = 3453


#============================================================================
# Functions
#============================================================================
def get_trainer(config_path):
    config = Params.from_file(config_path)
    config = ConfigParser(config, all=False, zero_shot=False, other_shot=False, predict=False)

    student = config.create_student(nocrf=False)
    corpus = config.corpus

    trainer = ReinforcementTrainer(student, None, corpus, config=config.config, **config.config["ReinforcementTrainer"], is_test=True)
    base_path = config.get_target_path

    trainer.embeddings_storage_mode = 'cpu'

    trainer.model.eval()
    trainer.model.to('cpu')

    trainer.model = trainer.model.load(base_path / "best-model.pt", device='cpu')
    training_state = torch.load(base_path / 'training_state.pt', map_location=torch.device('cpu'))
    trainer.best_action = training_state['best_action']
    trainer.model.selection = trainer.best_action

    for embedding in trainer.model.embeddings.embeddings:
        # manually fix the bug for the tokenizer becoming None
        if hasattr(embedding, 'tokenizer') and embedding.tokenizer is None:
            from transformers import AutoTokenizer
            name = embedding.name
            if '_v2doc' in name:
                name = name.replace('_v2doc', '')
            if '_extdoc' in name:
                name = name.replace('_extdoc', '')
            embedding.tokenizer = AutoTokenizer.from_pretrained(name, do_lower_case=True)
        if hasattr(embedding, 'model') and hasattr(embedding.model, 'encoder') and not hasattr(
                embedding.model.encoder,
                'config'):
            embedding.model.encoder.config = embedding.model.config
    return trainer


def get_loader(trainer):
    loader = ColumnDataLoader(list(trainer.corpus.test), args.batch_size,
                                   use_bert=trainer.use_bert,
                                   tokenizer=trainer.bert_tokenizer, model=trainer.model,
                                   sentence_level_batch=trainer.sentence_level_batch, sort_data=True)

    loader.assign_tags(trainer.model.tag_type, trainer.model.tag_dictionary)
    with torch.no_grad():
        log.info("-----------------------gpu_friendly_assign_embedding---------------------")
        trainer.gpu_friendly_assign_embedding([loader], selection=trainer.model.selection)
        if trainer.controller.model_structure is not None:
            trainer.assign_embedding_masks(loader, sample=False)
    return loader


def run_preprocess(args):
    if os.path.exists(os.path.join(args.pre_data_save_path, "sentence")) == False:
        os.mkdir(os.path.join(args.pre_data_save_path, "sentence"))
    if os.path.exists(os.path.join(args.pre_data_save_path, "lengths")) == False:
        os.mkdir(os.path.join(args.pre_data_save_path, "lengths"))
    longest_token_sequence_in_batch = 124

    trainer = get_trainer(args.config)

    loader = get_loader(trainer)
    log.info("---------------ace model starts preprocessing-----------------")
    batch_no = 0

    for batch in loader:
        sentences = batch

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        prediction_mode = False

        if prediction_mode and trainer.model.embedding_selector and trainer.model.use_rl and not trainer.model.use_gumbel:
            trainer.model.embeddings.embed(sentences, embedding_mask=trainer.model.selection)
        else:
            trainer.model.embeddings.embed(sentences)
        if hasattr(sentences, 'features'):
            if trainer.model.map_embeddings:
                new_list = []
                for idx, embedding_name in enumerate(sorted(sentences.features.keys())):
                    new_list.append(
                        trainer.model.map_linears[idx](sentences.features[embedding_name].to("cpu")))
                sentence_tensor = torch.cat(new_list, -1)
            elif trainer.model.embedding_selector:
                if trainer.model.use_rl:
                    if trainer.model.use_embedding_masks:
                        sentence_tensor = [sentences.features[x].to("cpu") for idx, x in
                                           enumerate(sorted(sentences.features.keys()))]
                        sentence_masks = [
                            torch.ones_like(sentence_tensor[idx]) * sentences.embedding_mask[:, idx, None,
                                                                    None].to("cpu") for idx, x in
                            enumerate(sorted(sentences.features.keys()))]
                        sentence_tensor = torch.cat(
                            [x * sentence_masks[idx] for idx, x in enumerate(sentence_tensor)], -1)
                    else:
                        if trainer.model.embedding_attention:
                            embatt = torch.sigmoid(trainer.model.selector)
                            sentence_tensor = torch.cat(
                                [sentences.features[x].to("cpu") * trainer.model.selection[idx] *
                                 embatt[idx]
                                 for idx, x in enumerate(sorted(sentences.features.keys()))], -1)
                        else:
                            sentence_tensor = torch.cat(
                                [sentences.features[x].to("cpu") * trainer.model.selection[idx] for
                                 idx, x in
                                 enumerate(sorted(sentences.features.keys()))], -1)
                else:
                    if trainer.model.use_gumbel:
                        if trainer.model.training:
                            selection = torch.nn.functional.gumbel_softmax(trainer.model.selector, hard=True)
                            sentence_tensor = torch.cat(
                                [sentences.features[x].to("cpu") * selection[idx][1] for idx, x in
                                 enumerate(sorted(sentences.features.keys()))], -1)
                        else:
                            selection = torch.argmax(trainer.model.selector, -1)
                            sentence_tensor = torch.cat(
                                [sentences.features[x].to("cpu") * selection[idx] for idx, x in
                                 enumerate(sorted(sentences.features.keys()))], -1)
                    else:
                        selection = torch.sigmoid(trainer.model.selector)
                        sentence_tensor = torch.cat(
                            [sentences.features[x].to("cpu") * selection[idx] for idx, x in
                             enumerate(sorted(sentences.features.keys()))], -1)

            else:
                sentence_tensor = torch.cat(
                    [sentences.features[x].to("cpu") for x in sorted(sentences.features.keys())], -1)

            if hasattr(trainer.model, 'keep_embedding'):
                if trainer.model.map_embeddings:
                    sentence_tensor = []
                    for idx, embedding_name in enumerate(sorted(sentences.features.keys())):
                        sentence_tensor.append(
                            trainer.model.map_linears[idx](
                                sentences.features[embedding_name].to("cpu")))
                else:
                    sentence_tensor = [sentences.features[x].to("cpu") for x in
                                       sorted(sentences.features.keys())]
                embedding_name = sorted(sentences.features.keys())[trainer.model.keep_embedding]

                if 'forward' in embedding_name or 'backward' in embedding_name:
                    for idx, x in enumerate(sorted(sentences.features.keys())):
                        if 'forward' not in x and 'backward' not in x:
                            sentence_tensor[idx].fill_(0)
                else:
                    for idx, x in enumerate(sorted(sentences.features.keys())):
                        if x != embedding_name:
                            sentence_tensor[idx].fill_(0)
                sentence_tensor = torch.cat(sentence_tensor, -1)
        else:
            # initialize zero-padded word embeddings tensor
            sentence_tensor = torch.zeros(
                [
                    len(sentences),
                    longest_token_sequence_in_batch,
                    trainer.model.embeddings.embedding_length,
                ],
                dtype=torch.float,
                device="cpu",
            )

            for s_id, sentence in enumerate(sentences):
                # fill values with word embeddings
                sentence_tensor[s_id][: len(sentence)] = torch.cat(
                    [token.get_embedding().unsqueeze(0) for token in sentence], 0
                )

        lengths_tensor = torch.tensor(lengths, dtype=torch.int32)

        if batch_no == len(loader)-1:
            number = len(loader[batch_no])
            sentence_temp = torch.ones(args.batch_size, sentence_tensor.shape[1], sentence_tensor.shape[2])
            sentence_temp[:number, :, :] = sentence_tensor
            sentence_temp[number:, :, :] = sentence_tensor[0, :, :]

            lengths_temp = torch.ones(args.batch_size, dtype=torch.int32)
            lengths_temp[:number] = lengths_tensor
            lengths_temp[number:] = lengths_tensor[0]
            sentence_tensor = sentence_temp
            lengths_tensor = lengths_temp

        sentence_np = np.asarray(sentence_tensor.cpu())
        lengths_np = np.asarray(lengths_tensor.cpu())
        sentence_np.tofile(
            os.path.join(args.pre_data_save_path, "sentence", "in_" + str(batch_no) + ".bin"))
        lengths_np.tofile(
            os.path.join(args.pre_data_save_path, "lengths", "in_" + str(batch_no) + ".bin"))

        log.info(f"save success {batch_no}")
        batch_no += 1


#============================================================================
# Main
#============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/doc_ner_best.yaml')
    parser.add_argument('--pre_data_save_path', default='./pre_data/')
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()
    if not os.path.exists(args.pre_data_save_path):
        os.makedirs(args.pre_data_save_path)

    run_preprocess(args)
