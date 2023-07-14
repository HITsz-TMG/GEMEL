# -*- coding: UTF-8 -*-



import argparse
import os


MODEL_PATH = {
    'opt-6.7b': 'facebook/opt-6.7b',
    'opt-2.7b': 'facebook/opt-2.7b',
    'opt-1.3b': 'facebook/opt-1.3b',
}


def get_args():
    args_parser = argparse.ArgumentParser(description='GEMEL')

    # dataset
    args_parser.add_argument('--dataset', type=str, default='wikidiverse', choices=['wikidiverse', 'wikimel'],)

    # dictionary or file
    args_parser.add_argument('--data_dir', type=str, default='./data/')
    args_parser.add_argument('--log_dir', type=str, default='./log/')
    args_parser.add_argument('--ckpt_dir', type=str, default='./checkpoint/')
    args_parser.add_argument('--cache_dir', type=str, default=None)
    args_parser.add_argument('--img_feat', type=str, default='clip_vit_large_patch14_1024.hdf5')
    args_parser.add_argument('--trie_file', type=str, default='prefix_tree_opt.pkl')
    args_parser.add_argument('--ment_embed_file', type=str, default='SimCSE_train_mention_embeddings.pkl')

    # model related
    args_parser.add_argument('--model_name', type=str, default='opt-6.7b')
    args_parser.add_argument('--max_new_tokens', type=int, default=32, help='max length of generation tokens')
    args_parser.add_argument('--num_beams', type=int, default=5)
    args_parser.add_argument('--use_prefix_tree', type=bool, default=True)
    args_parser.add_argument('--visual_prefix_length', type=int, default=4)
    args_parser.add_argument('--dim_clip', type=int, default=1024)
    args_parser.add_argument('--simcse_model', type=str, default='princeton-nlp/sup-simcse-roberta-large')

    # training related
    args_parser.add_argument('--train_bs', type=int, default=1)
    args_parser.add_argument('--random_seed', type=int, default=42)
    args_parser.add_argument('--lr', type=float, default=1e-6)
    args_parser.add_argument('--train_epoch', type=int, default=5)
    args_parser.add_argument('--weight_decay', type=float, default=0.01)
    args_parser.add_argument('--warmup_ratio', type=float, default=0.1)
    args_parser.add_argument("--max_norm", type=float, default=1.0, help='max norm of the gradients')
    args_parser.add_argument("--adam_epsilon", default=1e-6, type=float)
    args_parser.add_argument('--use_gradient_accumulation', type=bool, default=True)
    args_parser.add_argument('--accum_iter', type=int, default=16)
    args_parser.add_argument('--ICL_examples_num', type=int, default=16)

    # evaluation related
    args_parser.add_argument('--do_eval', type=bool, default=True)
    args_parser.add_argument('--eval_bs', type=int, default=1)
    args_parser.add_argument('--do_eval_steps', type=int, default=5000)

    # test
    args_parser.add_argument('--do_test', type=bool, default=True)
    # opt-6.7b_wikimel_linear_4token_16examples_75_53.pkl   opt-6.7b_wikidiverse_linear_4token_16examples_82_77.pkl
    args_parser.add_argument('--best_ckpt', type=str, default='opt-6.7b_wikidiverse_linear_4token_16examples_82_77.pkl') # for infe.py

    # parse
    args = args_parser.parse_args()

    # data
    args.dataset_dir = os.path.join(args.data_dir, args.dataset)
    args.data_file = {"train": os.path.join(args.dataset_dir, "train.json"),
                      "dev": os.path.join(args.dataset_dir, "dev.json"),
                      "test": os.path.join(args.dataset_dir, "test.json")}
    args.img_feat = os.path.join(args.dataset_dir, args.img_feat)
    args.trie_file = os.path.join(args.dataset_dir, args.trie_file)
    args.ment_embed_file = os.path.join(args.dataset_dir, args.ment_embed_file)

    args.model_path = MODEL_PATH[args.model_name]
    return args


if __name__ == '__main__':
    pass
