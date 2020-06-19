import os
from glob import glob
import struct
import logging
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from skimage import io, transform
from PIL import Image

def collate_fn_video(batch, padding=6):
    # batch.sort(key=lambda x: x['data'].shape[0], reverse=True)
    len_video = [x['data'].shape[0] for x in batch]
    len_label = [len(x['label']) for x in batch]
    batch_video = torch.zeros(len(len_video), max(len_video), batch[0]['data'].shape[1])
    batch_label = []
    IDs = []
    for i, bat in enumerate(batch):
        data = bat['data']
        label = bat['label']
        batch_label.extend(label)
        batch_video[i, :len_video[i], :] = torch.FloatTensor(data)
        IDs.append(bat['id'])
    batch_label = torch.LongTensor(batch_label)
    len_video = torch.LongTensor(len_video)
    len_label = torch.LongTensor(len_label)

    batch_video = batch_video.permute(0, 2, 1)

    # if padding != 0:
    #     downsample_rate = 4
    #     B, T, C, W, H = batch_video.shape
    #     left_pad = padding
    #     right_pad = (T // downsample_rate) * downsample_rate - T + padding
    #     batch_video_pad = torch.zeros([B, T + left_pad + right_pad, C, W, H]).float()
    #     batch_video_pad[:, left_pad:T+left_pad, :, :, :] = batch_video
    #     len_video += left_pad + right_pad
    # return {'data': batch_video_pad, 'label': batch_label, 'len_data': len_video, 'len_label': len_label, 'id': IDs}
    return {'data': batch_video, 'label': batch_label, 'len_data': len_video, 'len_label': len_label, 'id': IDs}

def collate_fn_clip(batch):
    IDs = []
    video_list = []
    label_list = []
    for i, bat in enumerate(batch):
        data = bat['data']
        label = bat['label']
        IDs.append(bat['id'])
        video_list.append(data)
        label_list.append(label)
    label_tensor = torch.LongTensor(label_list)
    video_tensor = torch.stack(video_list, dim=0)
    return  {'data': video_tensor, 'label': label_tensor, 'id': IDs}

class PhoenixVideo(Dataset):
    def __init__(self, corpus_dir, video_path, phase, DEBUG=False):
        self.vocab_file = './data/newtrainingClasses.txt'
        self.image_type = 'jpg'
        self.max_video_len = 10000
        self.corpus_dir = corpus_dir
        self.video_path = video_path
        self.phase = phase

        self.alignment = {}
        self.voc = Voc(self.vocab_file)

        self.phoenix_dataset = self.load_video_list()
        self.data_dict = self.phoenix_dataset[phase]
        if DEBUG == True:
            self.data_dict = self.data_dict[:101]
        logging.info('[DATASET: {:s}]: total {:d} samples.'.format(phase, len(self.data_dict)))

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        cur_vid_info = self.data_dict[idx]
        id = cur_vid_info['id']
        video_file = cur_vid_info['path']
        label = cur_vid_info['label']
        video_tensor = self.load_video(video_file)
        sample = {'id': id, 'data': video_tensor, 'label': label}
        return sample

    def load_video(self, video_name):
        feat = caffeFeatureLoader.loadVideoC3DFeature(video_name, 'pool5')
        feat = torch.tensor(feat)
        return feat

    def load_video1(self, video_name):
        frames_list = glob(os.path.join(video_name, '*.{:s}'.format(self.image_type)))
        frames_list.sort()
        num_frame = len(frames_list)
        if self.phase=='train' and self.sample and num_frame > self.max_video_len:
            for _ in range(num_frame-self.max_video_len):
                frames_list.pop(np.random.randint(len(frames_list)))
        frames_tensor_list = [self.load_image(frame_file) for frame_file in frames_list]
        video_tensor = torch.stack(frames_tensor_list, dim=0)
        return video_tensor

    def load_image(self, img_name):
        image = Image.open(img_name)
        image = self.transform(image)
        return image

    def load_video_list(self):
        phoenix_dataset = {}
        outliers = ['13April_2011_Wednesday_tagesschau_default-14'] # '05July_2010_Monday_heute_default-8'
        for task in ['train', 'dev', 'test']:
            if task != self.phase:
                continue
            dataset_path = os.path.join(self.video_path, task)
            corpus = pd.read_csv(os.path.join(self.corpus_dir, '{:s}.corpus.csv'.format(task)), sep='|')
            videonames = corpus['folder'].values
            annotation = corpus['annotation'].values
            ids = corpus['id'].values
            num_sample = len(ids)
            video_infos = []
            for i in range(num_sample):
                if ids[i] in outliers:
                    continue
                tmp_info = {
                    'id': ids[i],
                    'path': os.path.join(self.video_path, task, videonames[i].replace('*.png', '')),
                    'label_text': annotation[i],
                    'label': self.sentence2index(annotation[i])
                }
                video_infos.append(tmp_info)
            phoenix_dataset[task] = video_infos
        return phoenix_dataset

    def sentence2index(self, sent):
        sent = sent.split(' ')
        s = []
        for word in sent:
            if word in self.voc.word2index:
                s.append(self.voc.word2index[word])
            else:
                s.append(self.voc.word2index['<UNK>'])
        return s

class Voc():
    def __init__(self, vocab_file):
        PAD_token = 0
        self.vocab_file = vocab_file
        self.word2index = {'PAD': PAD_token}
        self.index2word = {PAD_token: 'PAD'}
        self.num_words = 1

        count = 0
        with open(self.vocab_file, 'r') as fid:
            for line in fid:
                if count != 0:
                    line = line.strip().split(' ')
                    word = line[0]
                    if word not in self.word2index:
                        self.word2index[word] = self.num_words
                        self.index2word[self.num_words] = word
                        self.num_words += 1
                count += 1
        UNK_token = self.num_words
        BOS_token = self.num_words + 1
        EOS_token = self.num_words + 2
        BLANK_token = self.num_words + 3
        self.word2index['<UNK>'] = UNK_token
        self.word2index['<BOS>'] = BOS_token
        self.word2index['<EOS>'] = EOS_token
        self.word2index['<BLANK>'] = BLANK_token
        self.index2word[UNK_token] = '<UNK>'
        self.index2word[BOS_token] = '<BOS>'
        self.index2word[EOS_token] = '<EOS>'
        self.index2word[BLANK_token] = '<BLANK>'
        self.num_words += 4

class caffeFeatureLoader():
    @staticmethod
    def loadVideoC3DFeature(sample_name, feattype = 'pool5'):
        featnames = glob(os.path.join(sample_name, '*.' + feattype))
        featnames.sort()
        feat = []
        for name in featnames:
            feat.append(caffeFeatureLoader.loadC3DFeature(name)[0])
        return feat

    @staticmethod
    def loadC3DFeature(filename):
        feat = []
        with open(filename, 'rb') as fileData:
            num = struct.unpack("i", fileData.read(4))[0]
            chanel = struct.unpack("i", fileData.read(4))[0]
            length = struct.unpack("i", fileData.read(4))[0]
            height = struct.unpack("i", fileData.read(4))[0]
            width = struct.unpack("i", fileData.read(4))[0]
            blob_shape = [num, chanel, length, height, width]
            m = num * chanel * length * height * width
            for i in range(m):
                val = struct.unpack("f", fileData.read(4))[0]
                feat.append(val)
        return feat, blob_shape


if __name__=='__main__':
    dl = PhoenixVideo(corpus_file='./data/train.corpus.csv',
        video_path='/home/pjh/data3/data/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px')
