# import debugpy;debugpy.connect(('10.119.54.66', 5627))
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import transformers
from transformers import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions
import math
import numpy as np
import pdb

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.dense = nn.Linear(in_dim, out_dim)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x) # non-linear activation

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

#follow the simcse's best performance, we only choose the 'cls' pooler type, which means [CLS] representation with BERT/RoBERTa's MLP pooler. And we code it in the follow codes, instead of defining a 'pooler' class.

# class Pooler(nn.Module):
#     """
#     Parameter-free poolers to get the sentence embedding
#     'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
#     'cls_before_pooler': [CLS] representation without the original MLP pooler.
#     'avg': average of the last layers' hidden states at each token.
#     'avg_top2': average of the last two layers.
#     'avg_first_last': average of the first and the last layers.
#     """
#     def __init__(self, pooler_type):
#         super().__init__()
#         self.pooler_type = pooler_type
#         assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

#     def forward(self, attention_mask, outputs):
#         last_hidden = outputs.last_hidden_state
#         pooler_output = outputs.pooler_output
#         hidden_states = outputs.hidden_states

#         if self.pooler_type in ['cls_before_pooler', 'cls']:
#             return last_hidden[:, 0]
#         elif self.pooler_type == "avg":
#             return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
#         elif self.pooler_type == "avg_first_last":
#             first_hidden = hidden_states[1]
#             last_hidden = hidden_states[-1]
#             pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
#             return pooled_result
#         elif self.pooler_type == "avg_top2":
#             second_last_hidden = hidden_states[-2]
#             last_hidden = hidden_states[-1]
#             pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
#             return pooled_result
#         else:
#             raise NotImplementedError



class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.args = model_kargs['model_args']
        self.bert = BertModel(config)
        self.pooler = MLPLayer(config.hidden_size, config.hidden_size)
        self.init_weights()

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        # flatten input for encoding
        input_ids = input_ids.view((-1, input_ids.size(-1))) #(bs * num_sent, len)
        attention_mask = attention_mask.view((-1,attention_mask.size(-1))) #(bs * num_sent, len)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) #(bs * num_sent, len)
        
        # print(inputs_embeds.device)


        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        pooler_output = self.pooler(outputs.last_hidden_state[:, 0])

        return BaseModelOutputWithPoolingAndCrossAttentions(
            pooler_output=pooler_output,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
        )

class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.args = model_kargs['model_args']
        self.roberta = RobertaModel(config)
        self.pooler = MLPLayer(config.hidden_size, config.hidden_size)
        self.init_weights()

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        # Flatten input for encoding
        input_ids = input_ids.view((-1, input_ids.size(-1)))  # (bs * num_sent, len)
        attention_mask = attention_mask.view((-1, attention_mask.size(-1)))  # (bs * num_sent len)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))  # (bs * num_sent, len)

        

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        pooler_output = self.pooler(outputs.last_hidden_state[:, 0])

        return BaseModelOutputWithPoolingAndCrossAttentions(
            pooler_output=pooler_output,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
        )


class CheatModel(nn.Module):
    pass

class CheatCSE(nn.Module):
    #Enhance the dimension of text embedding, simulating it as a picture and text pair
    def __init__(self, lang_model, cheat_model, args):
        super().__init__()
        self.args = args
        self.lang_model = lang_model
        self.cheat_model = cheat_model
        self.grounding = MLPLayer(args.hidden_size, args.proj_dim) # sent embeddings -> grounding space
        self.sim = Similarity(temp=self.args.temp)
        self.sim_cheat = Similarity(temp=self.args.temp_cheat)
        self.loss_fct = nn.CrossEntropyLoss()
    
    def forward(self, batch):
        lang_output = self.lang_model(input_ids=batch['input_ids'],
                                      attention_mask=batch['attention_mask'],
                                      token_type_ids=batch['token_type_ids'] if 'position_ids' in batch.keys() else None,
                                      position_ids=batch['position_ids'] if 'position_ids' in batch.keys() else None)


        batch_size = batch['input_ids'].size(0)
        num_sent = batch['input_ids'].size(1)

        #[bs*2, hidden] -> [bs, 2, hidden]
        lang_pooled_output = lang_output.last_hidden_state[:,0].view(batch_size, num_sent, -1)
        lang_projection = lang_output.pooler_output.view((batch_size, num_sent, -1)) # [bs, 2,  hidden],  output of additional MLP layer


        return lang_pooled_output, lang_projection

    def compute_loss(self, batch, cheat=False):
        l_pool, l_proj = self.forward(batch)

        # Separate representation
        z1, z2 = l_proj[:, 0], l_proj[:, 1]  # (bs, hidden)
        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))  # (bs, bs)

        labels = torch.arange(cos_sim.size(0)).long().to(self.args.device)  # [0, 1, bs-1]  (bs)
        loss = self.loss_fct(cos_sim, labels)  # unsup: bs-1 negatives

        if not cheat:
            return loss

        else:
            pass

        
        
