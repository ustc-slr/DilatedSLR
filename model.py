from tqdm import tqdm
import os
import logging
import time
import uuid

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader
import torchvision
import ctcdecode
from itertools import groupby

from lib.lib_data import PhoenixVideo, collate_fn_video
from lib.lib_metric import get_wer_delsubins
from evaluation_relaxation.phoenix_eval import get_phoenix_wer
from lib.utils import LossManager, ModelManager


class SLRNetwork(nn.Module):
    def __init__(self, opts, vocab_size, dilated_channels=512,
                 num_blocks=1, dilations=[1, 2, 4], dropout=0.0):
        super(SLRNetwork, self).__init__()
        self.opts = opts
        self.vocab_size = vocab_size
        self.in_channels = self.opts.feature_dim
        self.out_channels = dilated_channels

        self.num_blocks = num_blocks
        self.dilations = dilations
        self.kernel_size = 3

        self.block_list = nn.ModuleList()
        for i in range(self.num_blocks):
            self.block_list.append(DilatedBlock(self.in_channels, self.out_channels,
                                                self.kernel_size, self.dilations))
        self.out_conv = nn.Conv1d(self.out_channels, self.out_channels, self.kernel_size, padding=(self.kernel_size-1)//2)
        self.act_tanh = nn.Tanh()
        self.fc = nn.Linear(self.out_channels, self.vocab_size)

    def forward(self, video, len_video):
        out = 0
        for block in self.block_list:
            out += block(video)
        out = self.act_tanh(self.out_conv(out))
        logits = out.permute(0, 2, 1)
        logits = self.fc(logits)
        return logits

class DilatedCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(DilatedCell, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.in_channels = in_channels
        self.out_channels = out_channels

        stride = 1
        # padding = ((kernel_size-1)*(stride-1) + dilation*(kernel_size-1)) // 2
        padding = (kernel_size - 1) * dilation // 2
        self.in_conv = nn.Conv1d(self.in_channels, self.out_channels,
                                 self.kernel_size, dilation=self.dilation, padding=padding)
        self.mid_conv = nn.Conv1d(self.out_channels, self.out_channels,
                                 self.kernel_size, padding=(self.kernel_size-1)//2)
        self.gate_tanh = nn.Tanh()
        self.gate_sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''
        :param x: [B, C, L]
        :return:
        '''
        res = x
        x = self.in_conv(x)
        x = self.gate_tanh(x) * self.gate_sigmoid(x)
        o = self.gate_tanh(self.mid_conv(x))
        h = o + res
        return o, h

class DilatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_list):
        super(DilatedBlock, self).__init__()
        self.num_cells = len(dilation_list)
        self.dilated_cells = nn.ModuleList()
        self.dilated_cells.append(DilatedCell(in_channels, out_channels, kernel_size, dilation_list[0]))
        for dilation in dilation_list[1:]:
            self.dilated_cells.append(DilatedCell(out_channels, out_channels, kernel_size, dilation))

    def forward(self, x):
        block_o = 0
        for cell in self.dilated_cells:
            o, x = cell(x)
            block_o += o
        return block_o



class SLR(object):
    def __init__(self, opts, vocab_size, blank_id):
        self.opts = opts
        self.vocab_size = vocab_size
        self.blank_id = blank_id
        self.network = SLRNetwork(self.opts, vocab_size,
                                  num_blocks=5, dilations=[1, 2, 4])

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.CTCLoss(blank=self.blank_id, reduction='none')
        params_all = [{'params': self.network.parameters()}]
        self.optimizer = create_optimizer('adam', params_all, lr=self.opts.learning_rate,
                                          momentum=self.opts.momentum, weight_decay=self.opts.weight_decay)
        self.ctc_decoder_vocab = [chr(x) for x in range(20000, 20000 + self.vocab_size)]
        self.ctc_decoder = ctcdecode.CTCBeamDecoder(self.ctc_decoder_vocab, beam_width=self.opts.beam_width,
                                                    blank_id=self.blank_id, num_processes=10)
        self.decoded_dict = {}
        pass

    def create_dataloader(self, phase, batch_size, shuffle, num_workers=8, drop_last=False, DEBUG=False):
        dataset_phoenix = PhoenixVideo(corpus_dir=self.opts.corpus_dir, video_path=self.opts.video_path,
                                       phase=phase, DEBUG=DEBUG)
        dataloader = DataLoader(dataset_phoenix, batch_size=batch_size, shuffle=shuffle, pin_memory=True,
                                num_workers=num_workers, collate_fn=collate_fn_video, drop_last=drop_last)
        return dataloader

    def eval_batch(self, video, len_video, label, len_label, video_id, device=None):
        with torch.no_grad():
            bs = video.shape[0]
            video = video.cuda(non_blocking=True)
            len_video = len_video.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            len_label = len_label.cuda(non_blocking=True)

            self.network.eval()
            logits = self.network(video, len_video)  # logits.shape = [B, L, C]
            logits = F.softmax(logits, dim=-1)
            pred_seq, _, _, out_seq_len = self.ctc_decoder.decode(logits, len_video)

            err_delsubins = np.zeros([4])
            count = 0
            correct = 0
            start = 0
            for i, length in enumerate(len_label):
                end = start + length
                ref = label[start:end].tolist()
                hyp = [x[0] for x in groupby(pred_seq[i][0][:out_seq_len[i][0]].tolist())]
                self.decoded_dict[video_id[i]] = hyp
                correct += int(ref == hyp)
                err = get_wer_delsubins(ref, hyp)
                err_delsubins += np.array(err)
                count += 1
                start = end
            assert end == label.size(0)
        return err_delsubins, correct, count


    def train_batch(self, video, len_video, label, len_label, video_id, update_params, device=None, update_grad=True):
        '''
        :param video: [B, L, C]
        :param len_video: [L]
        :param label: [B, L]
        :param len_label: [B]
        :param device: [B]
        :param update_grad: True or False
        :return:
        '''
        bs = video.shape[0]
        video = video.cuda(non_blocking=True)
        len_video = len_video.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
        len_label = len_label.cuda(non_blocking=True)

        self.network.train()
        logits = self.network(video, len_video) # logits.shape = [B, L, C]
        logits = logits.permute(1, 0, 2)  # logits.shape = [L, B, C]
        log_probs = logits.log_softmax(-1)
        loss = self.criterion(log_probs, label, len_video, len_label)
        loss = loss.mean()

        loss.backward()
        if update_grad:
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss

    def train(self, pretrained_model=None):
        bs = self.opts.batch_size
        dataloader_train = self.create_dataloader(phase='train', batch_size=bs, num_workers=self.opts.data_worker,
                                                  shuffle=True, drop_last=False, DEBUG=self.opts.DEBUG)
        dataloader_dev = self.create_dataloader(phase='dev', batch_size=bs, num_workers=self.opts.data_worker,
                                                shuffle=False, drop_last=False, DEBUG=self.opts.DEBUG)
        self.dataloader_train = dataloader_train
        loss_manager = LossManager(print_step=50)
        model_manager = ModelManager(max_num_models=5)

        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            logging.info('Total use {:d} GPUs!]'.format(torch.cuda.device_count()))
            self.network = nn.DataParallel(self.network)
        else:
            logging.info('Only use 1 GPU!]')
        self.network.cuda()

        num_epoch = self.opts.num_epoch
        global_step = 0
        # optimizer.zero_grad()
        last_status = {'loss': -1., 'loss_trip': -1.}
        start_epoch = 0

        for epoch in range(start_epoch, num_epoch):
            # ---------- start of training section -----------
            # print('dataloader_train', len(dataloader_train))
            epoch_loss = []
            update_params = self.opts.update_param

            for i, item in tqdm(enumerate(dataloader_train), desc='[Training phase, epoch {:d}]'.format(epoch)):
                global_step += 1
                batch = item
                video_id = batch['id']
                video = batch['data']
                label = batch['label']
                len_video = batch['len_data']
                len_label = batch['len_label']
                update_flag = True if (global_step % self.opts.update_step) == 0 else False
                loss = self.train_batch(video, len_video, label, len_label, video_id, update_params, self.device, update_flag)
                loss = loss.item()
                loss_manager.update(loss, epoch, global_step)
                epoch_loss.append(loss)
            logging.info('Epoch: {:d}, loss: {:.3f} -> {:.3f}'.format(epoch, last_status['loss'], np.mean(epoch_loss)))
            last_status['loss'] = np.mean(epoch_loss)

            # ------------ Test ---------------
            val_err = np.zeros([4])
            val_correct = 0
            val_count = 0
            for i, item in tqdm(enumerate(dataloader_dev), desc='[Validation phase, epoch {:d}]'.format(epoch)):
                batch = item
                video_id = batch['id']
                video = batch['data']
                label = batch['label']
                len_video = batch['len_data']
                len_label = batch['len_label']

                err, correct, count = self.eval_batch(video, len_video, label, len_label, video_id, self.device)
                val_err += err
                val_correct += correct
                val_count += count
            logging.info('-' * 50)
            logging.info('DEV ACC: {:.5f}, {:d}/{:d}'.format(val_correct / val_count, val_correct, val_count))
            logging.info('DEV WER: {:.5f}, SUB: {:.5f}, INS: {:.5f}, DEL: {:.5f}'.format(\
                val_err[0] / val_count, val_err[1] / val_count, val_err[2] / val_count, val_err[3] / val_count))
            # ------ Evaluation with official script (merge synonyms) --------
            list_str_for_test = []
            for k, v in self.decoded_dict.items():
                start_time = 0
                for wi in v:
                    tl = np.random.random() * 0.1
                    list_str_for_test.append('{} 1 {:.3f} {:.3f} {}\n'.format(k, start_time, start_time + tl,
                                                                              dataloader_dev.dataset.voc.index2word[wi]))
                    start_time += tl
            tmp_prefix = str(uuid.uuid1())
            txt_file = '{:s}.txt'.format(tmp_prefix)
            result_file = os.path.join('evaluation_relaxation', txt_file)
            with open(result_file, 'w') as fid:
                fid.writelines(list_str_for_test)
            phoenix_eval_err = get_phoenix_wer(txt_file, 'dev', tmp_prefix)
            logging.info('[Relaxation Evaluation] DEV WER: {:.5f}, SUB: {:.5f}, INS: {:.5f}, DEL: {:.5f}'.format(\
                phoenix_eval_err[0], phoenix_eval_err[1], phoenix_eval_err[2], phoenix_eval_err[3]))

            model_name = os.path.join(self.opts.log_dir, 'ep{:d}.pkl'.format(epoch))
            if num_gpus > 1:
                torch.save(self.network.module.state_dict(), model_name)
            else:
                torch.save(self.network.state_dict(), model_name)
            model_manager.update(model_name, phoenix_eval_err, epoch)

    def test(self, model_file):
        phase = self.opts.eval_set
        logging.info('Restoring full model parameters from {:s}'.format(model_file))
        self.network.load_state_dict(torch.load(model_file), strict=True)
        num_gpu = len(''.join(self.opts.gpu.split()).split(','))
        bs = self.opts.batch_size
        dataloader_test = self.create_dataloader(phase=phase, batch_size=bs, num_workers=self.opts.data_worker,
                                                 shuffle=False, drop_last=False, DEBUG=self.opts.DEBUG)
        self.network.cuda()
        val_err = np.zeros([4])
        val_correct = 0
        val_count = 0
        t0 = time.time()
        t_data = 0
        t_total = 0
        t_gpu = 0
        for i, item in tqdm(enumerate(dataloader_test), desc='[Testing phase]'):
            t_data += time.time() - t0
            batch = item
            video_id = batch['id']
            video = batch['data']
            label = batch['label']
            len_video = batch['len_data']
            len_label = batch['len_label']
            t1 = time.time()
            err, correct, count = self.eval_batch(video, len_video, label, len_label, video_id, self.device)
            t_gpu += time.time() - t1
            val_err += err
            val_correct += correct
            val_count += count
            t_total += time.time() - t0
            t0 = time.time()
        logging.info('Total time: {:.3f}s, time for data: {:.3f}s, time for gpu: {:.3f}s, percentage: {:.3f}'.format(\
            t_total/val_count, t_data/val_count, t_gpu/val_count, t_data / t_total))
        logging.info('TEST ACC: {:.5f}, {:d}/{:d}'.format(val_correct / val_count, val_correct, val_count))
        logging.info('TEST WER: {:.5f}, SUB: {:.5f}, INS: {:.5f}, DEL: {:.5f}'.format(\
            val_err[0] / val_count, val_err[1] / val_count, val_err[2] / val_count, val_err[3] / val_count))

        list_str_for_test = []
        for k, v in self.decoded_dict.items():
            start_time = 0
            for wi in v:
                tl = np.random.random() * 0.1
                list_str_for_test.append('{} 1 {:.3f} {:.3f} {}\n'.format(k, start_time, start_time + tl,
                                                                          dataloader_test.dataset.voc.index2word[wi]))
                start_time += tl

        tmp_prefix = str(uuid.uuid1())
        txt_file = '{:s}.txt'.format(tmp_prefix)
        result_file = os.path.join('evaluation_relaxation', txt_file)
        with open(result_file, 'w') as fid:
            fid.writelines(list_str_for_test)
        phoenix_eval_err = get_phoenix_wer(txt_file, phase, tmp_prefix)
        logging.info('[Relaxation Evaluation] DEV WER: {:.5f}, SUB: {:.5f}, INS: {:.5f}, DEL: {:.5f}'.format( \
            phoenix_eval_err[0], phoenix_eval_err[1], phoenix_eval_err[2], phoenix_eval_err[3]))


def create_optimizer(optimizer, params, **kwargs):
    supported_optim = {
        'sgd': torch.optim.SGD, # momentum, weight_decay, lr
        'rmsprop': torch.optim.RMSprop, # momentum, weight_decay, lr
        'adam': torch.optim.Adam # weight_decay, lr
    }
    assert optimizer in supported_optim, 'Now only support {}'.format(supported_optim.keys())
    if optimizer == 'adam':
        del kwargs['momentum']
    optim = supported_optim[optimizer](params, **kwargs)
    logging.info('Create optimizer {}({})'.format(optimizer, kwargs))
    return optim

