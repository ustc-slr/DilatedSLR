import argparse
import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from model import SLRNetwork, SLR
from lib.utils import setup_seed, init_logging
from lib.lib_data import PhoenixVideo, collate_fn_video

# _DEBUG = False

def parse_args():
    p = argparse.ArgumentParser(description='SLR')
    p.add_argument('-t', '--task', type=str, default='train')
    p.add_argument('-g', '--gpu', type=str, default='0')

    # data
    p.add_argument('-dw', '--data_worker', type=int, default=8)
    p.add_argument('-fd', '--feature_dim', type=int, default=512)
    p.add_argument('-corp_dir', '--corpus_dir', type=str, default='./data')
    p.add_argument('-corp_tr', '--corpus_train', type=str, default='./data/train.corpus.csv')
    p.add_argument('-corp_te', '--corpus_test', type=str, default='./data/test.corpus.csv')
    p.add_argument('-corp_de', '--corpus_dev', type=str, default='./data/dev.corpus.csv')
    p.add_argument('-vp', '--video_path', type=str, default='/data2/pjh/data/feature/c3d_res_phoenix_body_iter5_120k')

    # optimizer
    p.add_argument('-op', '--optimizer', type=str, default='adam')
    p.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    p.add_argument('-wd', '--weight_decay', type=float, default=0.0)
    p.add_argument('-mt', '--momentum', type=float, default=0.9)
    p.add_argument('-nepoch', '--num_epoch', type=int, default=1000)
    p.add_argument('-us', '--update_step', type=int, default=1)
    p.add_argument('-upm', '--update_param', type=str, default='all')

    # train
    p.add_argument('-db', '--DEBUG', type=bool, default=False)
    p.add_argument('-lg_d', '--log_dir', type=str, default='./log/debug')
    p.add_argument('-bs', '--batch_size', type=int, default=1)
    p.add_argument('-ckpt', '--check_point', type=str, default='')

    # test (decoding)
    p.add_argument('-bwd', '--beam_width', type=int, default=5)
    p.add_argument('-vbs', '--valid_batch_size', type=int, default=1)
    p.add_argument('-evalset', '--eval_set', type=str, default='test', choices=['test', 'dev'])

    parameter = p.parse_args()
    return parameter

if __name__=='__main__':
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    setup_seed(8)
    opts = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu
    if not os.path.exists(opts.log_dir):
        print('Log DIR ({:s}) not exist! Make new folder.'.format(opts.log_dir))
        os.makedirs(opts.log_dir)
    init_logging(os.path.join(opts.log_dir, '{:s}_log.txt'.format(opts.task)))
    logging.info('Using random seed: 8')

    dataset_dev = PhoenixVideo(corpus_dir=opts.corpus_dir, video_path=opts.video_path, phase='dev', DEBUG=opts.DEBUG)
    vocab_size = dataset_dev.voc.num_words
    blank_id = dataset_dev.voc.word2index['<BLANK>']
    del dataset_dev

    slr = SLR(opts, vocab_size=vocab_size, blank_id=blank_id)
    if opts.task == 'train':
        logging.info(slr.network)
        logging.info(opts)
        if len(opts.check_point) == 0:
            slr.train()
        else:
            slr.train(opts.check_point)
    elif opts.task == 'test':
        slr.test(opts.check_point)
    else:
        pass