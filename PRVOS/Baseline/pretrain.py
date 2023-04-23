# -*- coding: utf-8 -*-
from __future__ import division

import argparse
from trainer import Trainer

import os, torch

import opts

def main():
    args = opts.get_args_parser()
    
    trainer = Trainer(args)
    trainer.cuda()
    trainer.set_trainset(args.dataset)
    trainer.set_valset(args.test_dataset)
    trainer.set_forwardset(args.dataset)
    
    if args.epoch != -1 or args.eval:
        trainer.load_model(args.epoch)  
    
    trainer.model.eval() # turn-off BN
    if args.eval or args.eval_first:
        if args.mode == 'eval_yv_forward' or 'train_yv':    # forward only on refer-yv-2019/train
            args.splits = 'train'
            trainer.evaluate()
        elif args.mode == 'eval_yv':    # evaluation on refer-yv-2019/valid
            args.splits = 'valid'
            args.mode = 'eval_yv'
            trainer.evaluate_refer_youtube_vos()
        elif args.mode == 'eval_davis':
            args.mode = 'eval_davis'
            trainer.evaluate()
        if not args.eval_first:
            return
    trainer.model.train()
    trainer.train()
    
if __name__ == "__main__":
    main()