# -*- coding: UTF-8 -*-
'''
=================================================
@Author : Senbao Shi
@Date   : 2023/7/13
@Desc   : model structure
=================================================
'''
import torch
from torch import nn


class GEMELModel(nn.Module):

    def __init__(self, lm, tokenizer, **kwargs):
        super(GEMELModel, self).__init__()
        self.lm = lm
        self.tokenizer = tokenizer
        self.kwargs = kwargs
        self.text_embedder = self.lm.model.decoder.embed_tokens  # for opt
        self.linear = nn.Linear(kwargs['dim_clip'], kwargs['dim_embedding'] * kwargs['visual_prefix_length'], dtype=torch.float16)

    def forward(self, batch_pairs, batch_targets):
        inputs_embeds, attention_mask, labels = self._concat_image_text_embeddings_train(batch_pairs, batch_targets)
        outputs = self.lm(input_ids=None, inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        return outputs

    def generate(self, batch_pairs, **kwargs):
        inputs_embeds, attention_mask = self._concat_image_text_embeddings_dev(batch_pairs)
        genenrated = self.lm.generate(input_ids=None, inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)
        return genenrated

    def _concat_image_text_embeddings_train(self, batch_pairs, batch_targets):
        batch_embeds, batch_labels = [], []
        for i in range(len(batch_pairs)): # bs
            item_feat = []
            for pair in batch_pairs[i]: # examples + item
                img_feat, text = pair
                # image
                img_feat_ = self.linear(torch.tensor(img_feat, dtype=torch.float16).to(self.kwargs['device']))  # (prefix_length *  dim_embedding)
                img_projected_embed = torch.reshape(img_feat_, (self.kwargs['visual_prefix_length'], self.kwargs['dim_embedding']))  # (prefix_length, dim_embedding)
                item_feat.append(img_projected_embed)
                # text
                text_ids = self.tokenizer(text, return_tensors="pt").input_ids[0, 1:].to(self.kwargs['device']) # (token_num) without bos </s>
                text_embed = self.text_embedder(text_ids)  # (token_num, dim_embedding)
                item_feat.append(text_embed)

            # target
            target_ids = self.tokenizer(batch_targets[i] + '</s>', return_tensors="pt").input_ids[0, 1:].to(self.kwargs['device'])  # (token_num) remove bos, add eos
            target_embed = self.text_embedder(target_ids)  # (token_num, dim_embedding)
            item_feat.append(target_embed)
            item_embeds = torch.cat(item_feat, dim=0) # (token_num, dim_embedding)
            batch_embeds.append(item_embeds)

            # label
            before_label_ids = torch.zeros(item_embeds.shape[0] - target_ids.shape[0], dtype=torch.int64, device=self.kwargs['device']) - 100 # -100 means not to calculate loss
            target_ids_ = torch.cat([before_label_ids, target_ids], dim=0)  # (token_num)
            batch_labels.append(target_ids_)  # (token_num)

        # train: right padding for batch
        tokens_length = [item.shape[0] for item in batch_embeds]
        max_token_len = max(tokens_length)

        pad_ids = self.tokenizer('<pad>', return_tensors="pt")['input_ids'][0, 1:].to(self.kwargs['device'])
        pad_embed = self.text_embedder(pad_ids)  # (1, dim_embedding)

        inputs_embeds, labels, attention_mask = [], [], []
        for i in range(len(batch_embeds)):
            item_embeds_, item_label = batch_embeds[i], batch_labels[i]

            if item_embeds_.shape[0] < max_token_len: # need to pad right
                pad_num = max_token_len - item_embeds_.shape[0]
                # pad embedding
                pad_item_embeds = torch.cat([item_embeds_] + [pad_embed] * pad_num, dim=0) # (max_token_len, dim_embedding)
                inputs_embeds.append(pad_item_embeds)
                # pad label
                pad_label_ids = torch.zeros(pad_num, dtype=torch.int64, device=self.kwargs['device']) - 100
                pad_item_label = torch.cat([item_label, pad_label_ids], dim=0) # (max_token_len)
                labels.append(pad_item_label)
                # pad attention_mask
                item_attention = [1, ] * item_embeds_.shape[0] + [0, ] * pad_num
                attention_mask.append(torch.tensor(item_attention, dtype=torch.int64, device=self.kwargs['device']))
            else:
                inputs_embeds.append(item_embeds_)
                labels.append(item_label)
                attention_mask.append(torch.ones(max_token_len, dtype=torch.int64, device=self.kwargs['device']))
        return torch.stack(inputs_embeds, dim=0), torch.stack(attention_mask, dim=0), torch.stack(labels, dim=0)# (b, max_token_len, dim_embedding), (b, max_token_len), (b, max_token_len)

    def _concat_image_text_embeddings_dev(self, batch_pairs):
        batch_embeds = []
        for pairs in batch_pairs: # bs
            item_feat = []

            for pair in pairs:
                img_feat, text = pair
                # image
                img_feat_ = self.linear(torch.tensor(img_feat, dtype=torch.float16).to(self.kwargs['device']))  # (prefix_length *  dim_embedding)
                img_projected_embed = torch.reshape(img_feat_, (self.kwargs['visual_prefix_length'], self.kwargs['dim_embedding']))  # (prefix_length, dim_embedding)
                item_feat.append(img_projected_embed)
                # text
                text_ids = self.tokenizer(text, return_tensors="pt").input_ids[0, 1:].to(self.kwargs['device']) # (token_num) without </s>
                text_embed = self.text_embedder(text_ids)  # (token_num, dim_embedding)
                item_feat.append(text_embed)
            item_embeds = torch.cat(item_feat, dim=0)  # (token_num, dim_embedding)
            batch_embeds.append(item_embeds)

        # dev: left padding
        tokens_length = [item.shape[0] for item in batch_embeds]
        max_token_len = max(tokens_length)

        pad_ids = self.tokenizer('<pad>', return_tensors="pt")['input_ids'][0, 1:].to(self.kwargs['device'])
        pad_embed = self.text_embedder(pad_ids)  # (1, dim_embedding)

        inputs_embeds, attention_mask = [], []
        for item in batch_embeds:
            if item.shape[0] < max_token_len: # need to pad left
                pad_num = max_token_len - item.shape[0]
                # pad embedding  left padding
                pad_item_embeds = torch.cat([pad_embed] * pad_num + [item], dim=0) # (max_token_len, dim_embedding)
                inputs_embeds.append(pad_item_embeds)
                # pad attention_mask  left padding
                item_attention = [0, ] * pad_num + [1, ] * item.shape[0]
                attention_mask.append(torch.tensor(item_attention, dtype=torch.int64, device=self.kwargs['device']))
            else:
                inputs_embeds.append(item)
                attention_mask.append(torch.ones(max_token_len, dtype=torch.int64, device=self.kwargs['device']))
        return torch.stack(inputs_embeds, dim=0), torch.stack(attention_mask, dim=0) # (b, max_token_len, dim_embedding)




if __name__ == '__main__':
    pass
