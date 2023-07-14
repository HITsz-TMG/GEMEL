# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# https://github.com/facebookresearch/GENRE/tree/main/genre
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
    we rectify the method 'get', make sure prefix_sequence start with eos
    when inference with input_embedding, prefix_sequence always start with eos
'''

from typing import Dict, List


class Trie(object):
    def __init__(self, sequences: List[List[int]] = [], end_token_id: int =None):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for sequence in sequences:
                Trie._add_to_trie(sequence, self.trie_dict)
                self.len += 1

        self.append_trie = None
        self.bos_token_id = None
        self.end_token_id = end_token_id

    def append(self, trie, bos_token_id):
        self.append_trie = trie
        self.bos_token_id = bos_token_id

    def add(self, sequence: List[int]):
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: List[int]):
        '''rectify the prefix_sequence'''
        prefix_sequence_ = []
        for i in range(len(prefix_sequence) - 1, -1, -1):
            if prefix_sequence[i] == self.end_token_id: # When passing embeddings to LLM, the sequence begins with eos
                prefix_sequence_ = prefix_sequence[i:]
                break
        if len(prefix_sequence_) == 0:
            raise Exception('do not have start token')
        return Trie._get_from_trie(
            prefix_sequence_, self.trie_dict, self.append_trie, self.bos_token_id
        )

    @staticmethod
    def load_from_dict(trie_dict, end_token_id):
        trie = Trie(end_token_id=end_token_id)
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie

    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(
        prefix_sequence: List[int],
        trie_dict: Dict,
        append_trie=None,
        bos_token_id: int = None,
    ):
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            if append_trie and bos_token_id in output:
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                trie_dict[prefix_sequence[0]],
                append_trie,
                bos_token_id,
            )
        else:
            if append_trie:
                return append_trie.get(prefix_sequence)
            else:
                return []

    def __iter__(self):
        def _traverse(prefix_sequence, trie_dict):
            if trie_dict:
                for next_token in trie_dict:
                    yield from _traverse(
                        prefix_sequence + [next_token], trie_dict[next_token]
                    )
            else:
                yield prefix_sequence

        return _traverse([], self.trie_dict)

    def __len__(self):
        return self.len

    def __getitem__(self, value):
        return self.get(value)





if __name__ == '__main__':
    # tree = Trie([[1, 2, 3], [1, 5, 6]])
    # print(tree.get([1]))
    pass
