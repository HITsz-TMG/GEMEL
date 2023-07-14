# -*- coding: utf-8 -*-
'''
=================================================
@Author : Senbao Shi
@Date   : 2023/7/13
@Desc   : Train the GEMEL model and save the optimal checkpoint.
          Parameters are set in params.py. The model structure is in model.py
=================================================
'''



import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel

from model import GEMELModel

import random

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import params
from utils import check_dirs, set_seed, GEMELDataset, calc_acc, load_prefix_tree, get_embed, train_configure





def _test(args):
    test_ds = GEMELDataset(args.data_file['test'], args.tokenizer, **args.kwargs_ds)
    test_dl = DataLoader(test_ds, batch_size=args.eval_bs, collate_fn=test_ds.collate_fn, shuffle=False)

    print('\nload linear')
    checkpoint_file = f'{args.ckpt_dir}{args.model_name}_{args.dataset}_linear_{args.visual_prefix_length}token_{args.ICL_examples_num}examples.pkl'  # only save 1 checkpoint
    checkpoint = torch.load(checkpoint_file)
    args.model.linear.load_state_dict(checkpoint)

    acc = _eval(args, test_dl)


def _eval(args, dl):
    print('Test...')
    # record
    eval_steps = len(dl)
    predictions, targets = [], []
    # evaluate
    args.model.eval()
    with torch.no_grad():
        with tqdm(total=eval_steps) as pbar:
            for step, batch_data in enumerate(dl):
                pbar.set_description(f'eval steps: {step}')
                pbar.update(1)

                batch_pairs, batch_targets = batch_data
                features = {
                    "batch_pairs": batch_pairs,
                    "num_beams": args.num_beams,
                    "num_return_sequences": 1,
                    "max_new_tokens": args.max_new_tokens,
                }
                if args.use_prefix_tree:
                    features['prefix_allowed_tokens_fn'] = lambda batch_id, sent: args.trie.get(sent.tolist())
                generated = args.model.generate(**features)
                batch_preds = args.tokenizer.batch_decode(generated, skip_special_tokens=True)
                predictions.extend(batch_preds)
                targets.extend(batch_targets)
                # log prediction
                if not step % 100:
                    i = random.randint(0, len(batch_targets) - 1)
                    input_text = ''.join([t for _, t in batch_pairs[i][-4:]])
                    print(f'\ninput_text:\n{input_text}')
                    print(f'\nresult: {batch_targets[i]==batch_preds[i].strip(" ")}\t\ttarget: {batch_targets[i]}\t\tpred: {batch_preds[i]}')
    acc = calc_acc(predictions, targets)
    return acc


def _eval2save(args):
    acc = _eval(args, args.eval_dl)
    args.writer.add_scalar('eval_acc', acc, args.global_steps)
    # judge to save
    if acc >= args.best_eval_acc:
        print(f'\nNew best model, new acc {acc:.4f} % >= previous acc {args.best_eval_acc:.4f} %')
        args.best_eval_acc = acc
        checkpoint_file = f'{args.ckpt_dir}{args.model_name}_{args.dataset}_linear_{args.visual_prefix_length}token_{args.ICL_examples_num}examples.pkl'  # only save 1 checkpoint
        torch.save(args.model.linear.state_dict(), checkpoint_file)
        print(f'\nSave to {checkpoint_file}')

    elif acc < args.best_eval_acc:
        print(f'\ndo not save, best acc: {args.best_eval_acc:.4f}')


def _train(args):
    print('Training...')

    # 1.record: loss steps
    args.writer = SummaryWriter(args.log_dir)
    ls_sum = 0.0
    args.global_steps = 0

    # 2.train and evaluate
    with tqdm(total=args.total_steps) as pbar:
        for epoch in range(args.train_epoch):
            args.epoch = epoch
            for batch_idx, batch_data in enumerate(args.train_dl):
                args.global_steps += 1

                args.model.train()
                batch_pairs, batch_targets = batch_data
                ls = args.model(batch_pairs, batch_targets).loss
                if args.use_gradient_accumulation:
                    ls /= args.accum_iter
                ls.backward()

                if args.use_gradient_accumulation:
                    if ((batch_idx + 1) % args.accum_iter == 0) or (batch_idx + 1 == len(args.train_dl)):
                        torch.nn.utils.clip_grad_norm_(args.model.parameters(), args.max_norm)
                        args.optimizer.step()
                        args.scheduler.step()
                        args.optimizer.zero_grad()
                else:
                    torch.nn.utils.clip_grad_norm_(args.model.parameters(), args.max_norm)
                    args.optimizer.step()
                    args.scheduler.step()
                    args.optimizer.zero_grad()

                ls_ = ls.item()
                ls_sum += ls_
                # 1) record
                args.writer.add_scalar('train_ls', ls_sum / args.global_steps, args.global_steps)
                pbar.set_description(f'ep: {epoch}, steps: {args.global_steps}, ls: {ls_:.4f}')
                pbar.update(1)
                # 2) evaluate and save
                if args.do_eval and not args.global_steps % args.do_eval_steps:
                    _eval2save(args)


def _main(args):
    # 1.check dir
    check_dirs(dirs=[args.dataset_dir, args.log_dir, args.ckpt_dir])

    # 2.random seed
    set_seed(args.random_seed)

    # 3.model and tokenizer
    from transformers import AutoTokenizer, OPTForCausalLM
    args.tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #  load the model in half-precision to accelerate generation and optimize memory consumption on GPU
    lm = OPTForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, cache_dir=args.cache_dir)
    # freeze large language model
    print('\nFreeze LLM\n')
    for param in lm.parameters():
        param.requires_grad = False

    args.dim_embedding = lm.config.hidden_size
    kwargs_model = {'dim_clip': args.dim_clip, 'dim_embedding': args.dim_embedding,
                    'visual_prefix_length': args.visual_prefix_length, 'device': args.device}
    args.model = GEMELModel(lm=lm, tokenizer=args.tokenizer, **kwargs_model).to(args.device)

    # model for calculating similarity
    args.train_embed = get_embed(args.ment_embed_file)
    args.roberta_tokenizer = AutoTokenizer.from_pretrained(args.simcse_model)
    args.roberta_model = AutoModel.from_pretrained(args.simcse_model, cache_dir=args.cache_dir).to(args.device)

    # 4.data
    # train dataset for calculating ICL similarity
    args.ICL_ds = GEMELDataset(args.data_file['train'], tokenizer=None, img_feat=args.img_feat)

    args.kwargs_ds = {'train_ds': args.ICL_ds, 'ICL_examples_num': args.ICL_examples_num, 'img_feat': args.img_feat, 'device': args.device,
                      'train_embed': args.train_embed, 'roberta_tokenizer': args.roberta_tokenizer, 'roberta_model': args.roberta_model}
    train_ds = GEMELDataset(args.data_file['train'], args.tokenizer, train_flag=True, **args.kwargs_ds)  # train_flag: exclude same training example
    dev_ds = GEMELDataset(args.data_file['dev'], args.tokenizer, **args.kwargs_ds)
    print(f'\ntrain data num: {len(train_ds)}  dev data num: {len(dev_ds)}')

    args.train_dl = DataLoader(dataset=train_ds, batch_size=args.train_bs, collate_fn=train_ds.collate_fn, shuffle=True)
    args.eval_dl = DataLoader(dataset=dev_ds, batch_size=args.eval_bs, collate_fn=dev_ds.collate_fn, shuffle=False)
    args.total_steps = args.train_epoch * len(args.train_dl)

    # prefix tree
    args.trie = load_prefix_tree(args.trie_file, args.tokenizer.eos_token_id) if args.use_prefix_tree else None

    # 5.train
    args.optimizer, args.scheduler = train_configure(args)
    args.best_eval_acc = float('-inf')
    _train(args)
    _eval2save(args)

    # 6.inference test
    if args.do_test: _test(args)


if __name__ == '__main__':
    args = params.get_args()
    print(args)

    _main(args)

