# Author Toshihiko Aoki
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
"""Bert SWEM"""


import numpy as np
import torch
from .models.bert import Config, BertModel
from .models.albert import AlbertModel
from .models.embed_projection_albert import EmbedProjectionAlbertModel
from .utils import get_tokenizer, load, to_bert_ids


class BertSWEM(object):

    def __init__(
        self,
        config_path='../config/bert_base.json',
        max_pos=-1,
        vocab_path=None,
        tokenizer_name='google',
        sp_model_path=None,
        bert_model_path=None,
        device='cpu',
        model_name='bert',
        encoder_json_path=None,
        vocab_bpe_path=None
    ):
        if vocab_path is None or bert_model_path is None:
            raise ValueError('Require vocab_path and bert_model_path')

        self.tokenizer = get_tokenizer(
            vocab_path=vocab_path, sp_model_path=sp_model_path, name=tokenizer_name,
            encoder_json_path=encoder_json_path, vocab_bpe_path=vocab_bpe_path)

        config = Config.from_json(config_path, len(self.tokenizer), max_pos)
        print(config)
        if model_name == 'proj':
            self.bert_model = EmbedProjectionAlbertModel(config)
        elif model_name == 'albert':
            self.bert_model = AlbertModel(config)
        else:
            self.bert_model = BertModel(config)
        self.max_pos = config.max_position_embeddings
        load(self.bert_model, bert_model_path, device)
        super().__init__()

    def embedding_vector(
        self,
        text=None,
        pooling_layer=-1,
        pooling_strategy='REDUCE_MEAN',
        hier_pool_window=-1
    ):
        ids = to_bert_ids(max_pos=self.max_pos, tokenizer=self.tokenizer, sentence_a=text)
        tensor = [torch.tensor([x], dtype=torch.long) for x in ids]

        self.bert_model.eval()
        with torch.no_grad():
            hidden_states, _ = self.bert_model(
                input_ids=tensor[0], token_type_ids=tensor[1], attention_mask=tensor[2], layer=pooling_layer)

        embedding = hidden_states.cpu().numpy()[0]
        if pooling_strategy == "REDUCE_MEAN":
            return np.mean(embedding, axis=0)
        elif pooling_strategy == "REDUCE_MAX":
            return np.max(embedding, axis=0)
        elif pooling_strategy == "REDUCE_MEAN_MAX":
            return np.r_[np.max(embedding, axis=0), np.mean(embedding, axis=0)]
        elif pooling_strategy == "CLS_TOKEN":
            return embedding[0]
        elif pooling_strategy == 'HIER':
            text_len = embedding.shape[0]
            if hier_pool_window > text_len or hier_pool_window < 1:
                hier_pool_window = text_len
            window_average_pooling = [np.mean(embedding[i:i + hier_pool_window], axis=0)
                                      for i in range(text_len - hier_pool_window + 1)]
            return np.max(window_average_pooling, axis=0)
        else:
            raise ValueError("Support pooling_strategy: {REDUCE_MEAN, REDUCE_MAX, REDUCE_MEAN_MAX, CLS_TOKEN, HIER}")

