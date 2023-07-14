# -*- coding: UTF-8 -*-
'''
=================================================
@Author : Senbao Shi
@Date   : 2023/7/13
@Desc   : dataset and some processing functions
=================================================
'''
import os
import h5py
from torch.optim import AdamW

import json
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics.pairwise import cosine_similarity

from trie import Trie


class GEMELDataset(Dataset):

    def __init__(self, file, tokenizer, **kwargs):
        self.file = file
        self.data = self._get_data(file)

        self.tokenizer = tokenizer
        self.kwargs = kwargs

        self._add_img_feat() # convert image url to clip output feature
        if tokenizer: # ICL dataset does not need tokenizer and examples
            self._get_examples() # get ICL examples from training set

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def _get_data(self, file):
        # load file and fiter NIL/OOV
        with open(file, encoding="utf-8") as f:
            raw_data = [json.loads(line) for line in f]
        data = [raw_data[i] for i in range(len(raw_data)) if raw_data[i]['golden'] != "NIL"] # remove NIL/OOV
        file_name = file.split('/')[-1]
        print(f'{file_name}\t\traw data num: {len(raw_data)}\t\tprocessed data num: {len(data)}')
        return data

    def _add_img_feat(self):
        # convert image url to clip output feature
        with h5py.File(self.kwargs['img_feat'], 'r') as h:
            for tmpDict in self.data:
                if 'img_name' in tmpDict: # wikimel dataset
                    img_url = tmpDict['img_name']
                else: # wikidiverse dataset
                    img_url = tmpDict["img_url"]
                try:
                    tmpDict["image"] = h[img_url][()]  # add image attribution to self.data
                except:
                    raise Exception(f'\ncan not load {img_url} from {self.kwargs["img_feat"]}')

    def _get_examples(self):
        print(f'\nRetrieve {self.file} ICL examples')
        for tmpDict in tqdm(self.data):
            mention = tmpDict['mention']
            text = tmpDict['text']

            query = self.kwargs['roberta_tokenizer'](mention, padding=True, truncation=True, return_tensors="pt").to(self.kwargs['device'])
            with torch.no_grad():
                query_embed = self.kwargs['roberta_model'](**query, output_hidden_states=True, return_dict=True).pooler_output
            scores = cosine_similarity(self.kwargs['train_embed'], query_embed.cpu().numpy())
            scores_ = scores.squeeze(-1)

            i = self.kwargs['ICL_examples_num']
            index_list = np.argsort(scores_).tolist()  # similarity index(from low to high)
            if self.kwargs and 'train_flag' in self.kwargs.keys(): # train, need to exclude item itself
                train_example_cnt = 0
                tmp_index = -1  # from right to left  -1 -2 -3
                prefix_items = []
                while train_example_cnt < i:
                    train_index = index_list[tmp_index]
                    item = self.kwargs['train_ds'][train_index]
                    tmp_index -= 1
                    if item['mention'] == mention and item['text'] == text:  # the same as train item
                        continue
                    else:  # different with train item
                        prefix_items.append(item)  # similarity decreases from left to right
                        train_example_cnt += 1
                prefix_items.reverse()  # similarity rises from left to right
            else:  # dev or test
                sorted_index = index_list[-i:]  # from low to high, select ICL_examples_num items
                prefix_items = [self.kwargs['train_ds'][index] for index in sorted_index]

            tmpDict['examples'] = prefix_items


    def collate_fn(self, items):
        batch_pairs, batch_targets = [], []
        for item in items:
            batch_pairs.append(self._get_pairs(item))
            batch_targets.append(item['target'])
        return batch_pairs, batch_targets

    def _get_pairs(self, item):
        text, mention, image = item['text'], item['mention'], item['image']
        if self.kwargs['ICL_examples_num'] != 0:
            prefix_list = self._similar_prefix(item)
        else:
            prefix_list = []
        text_ = f'[Text]{text}\n[Question]What does {mention} mentioned in the text refer to?\n[Answer]'
        pair = (image, text_)
        pair_list = prefix_list + [pair]
        return pair_list

    def _similar_prefix(self, item):
        prefix_items = item['examples']
        prefix_list = []
        for demo in prefix_items:
            text_ = f'[Text]{demo["text"]}\n[Question]What does {demo["mention"]} mentioned in the text refer to?\n[Answer]{demo["target"]}\n'
            image = demo['image']
            prefix_list.append((image, text_))
        return prefix_list


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def check_dirs(dirs=[]):
    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)
            print(f'Create directory: {dir}')
        else:
            print(f'Existed: {dir}')


def calc_acc(predictions, targets):
    assert len(predictions) == len(targets)
    hits = [predictions[i].strip(" ") == targets[i] for i in range(len(targets))]
    acc = 100.0 * sum(hits) / len(hits)
    print(f'\nacc: {acc:.4f} %')
    return acc


def load_prefix_tree(trie_file, eos_token_id):
    print(f'\nload prefix tree')
    trie_dict = pd.read_pickle(trie_file)
    trie = Trie.load_from_dict(trie_dict, eos_token_id)
    print(f'\ndone\n')
    return trie


def get_embed(file_path):
    import pickle
    with open(file_path, 'rb') as f:
        all_embeds = pickle.load(f)
    return all_embeds


def train_configure(args):
    # https://github.com/google-research/bert/blob/master/optimization.py#L25
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in args.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in args.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    warmup_steps = int(args.total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=args.total_steps)
    return optimizer, scheduler


if __name__ == '__main__':


    pass
