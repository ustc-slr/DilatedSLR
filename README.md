# DilatedSLR
This is the PyTorch reimplementation of DilatedSLR (IJCAI'18). For technical details, please refer to:
**Dilated Convolutional Network with Iterative Optimization for Coutinuous Sign Language Recognition** [[Paper](https://www.ijcai.org/Proceedings/2018/123)]

The results of this implementation may be slightly different from the original performance reported in the paper. 
The results in our paper are obtained using the TensorFlow implementation.

If it helps your research, please consider citing the following paper in your publications:
```bibtex
@inproceedings{pu2018dilated,
  title={Dilated Convolutional Network with Iterative Optimization for Coutinuous Sign Language Recognition},
  author={Pu, Junfu and Zhou, Wengang and Li, Houqiang},
  booktitle={International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2018}
}
```

## Table of Contents
- [Preparation](#preparation)
  * [Requirements](#requirements)
    + [1. Install](#1-install)
    + [2. Docker](#2-docker)
      - [Build with Dockerfile](#build-with-dockerfile)
      - [Use pre-built Docker image](#use-pre-built-docker-image)
  * [Dataset](#dataset)
- [Run](#run)
  * [Train](#train)
  * [Test](#test)
- [Related Resources](#related-resources)
- [Contact](#contact)

## Preparation
### Requirements
#### 1. Install
All required python packages are included in `requirements.txt`.
You can install the packages by
```bash
pip install -r requirements.txt
```
Please install *ctcdecode* following the official [instruction](https://github.com/parlance/ctcdecode).
If the installation failed with the problem of network connection, please consider installing by
```bash
git clone --recursive https://github.com/Jevin754/ctcdecode.git
cd ctcdecode
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple .
```

When calculating the evaluation metric with the official script provided by RWTH-PHOENIX-Weather, *SCKT* toolkit is required, one can install the toolkit with the following commands
```bash
git clone --recursive https://github.com/usnistgov/SCTK.git
cd SCTK
make config && make all && make check && make install && make doc
# add SCTK path to environment variable PATH
export PATH=$PATH:SCTK_PATH/bin/sclite
```
For more details about SCTK, please refer to [SCTK repository](https://github.com/usnistgov/SCTK).

#### 2. Docker
Configuring the environment is complicated. Hence, running with Docker is recommended. We provide Dockerfile and image for easy getting start.
##### Build with Dockerfile
To build Docker image from the source, run the following commands
```bash
git clone --recursive https://github.com/ustc-slr/DilatedSLR.git
cd DilatedSLR
docker build --no-cache -f ./Dockerfile -t ustcslr .
```
##### Use pre-built Docker image
To run with pre-built docker image, install Docker and start a new container with the docker image via
```bash
docker run --runtime=nvidia --rm -it --name ustcslr/ustcslr:latest
```

### Dataset
All experiments are performed on RWTH-PHOENIX-Weather-2014 (Multisigner). 
The image/video features are extracted with C3D.
We have released the C3D features.
Please download the features with the following link:
- C3D Feature [[Google Drive](https://drive.google.com/file/d/1y_xaNCjMJdLzE4PQCdTt3Ff4vt2FGy_R/view?usp=sharing)] [[Baidu Drive](https://pan.baidu.com/s/1lNxPADvUyJ2jg-r5JBZtEw) (pwd: 5w9f)] [[Rec](https://rec.ustc.edu.cn/share/0032cb50-b229-11ea-9a10-7bcdbb495df0)]

Extract features via
```bash
tar -zxvf c3d_res_phoenix_body_iter5_120k.tar.gz
```

## Run
The training and testing functions are both included in `main.py`. The parameters are as follows,
```bash
python main.py -h
usage: main.py [-h] [-t TASK] [-g GPU] [-dw DATA_WORKER] [-fd FEATURE_DIM]
               [-corp_dir CORPUS_DIR] [-corp_tr CORPUS_TRAIN]
               [-corp_te CORPUS_TEST] [-corp_de CORPUS_DEV] [-vp VIDEO_PATH]
               [-op OPTIMIZER] [-lr LEARNING_RATE] [-wd WEIGHT_DECAY]
               [-mt MOMENTUM] [-nepoch NUM_EPOCH] [-us UPDATE_STEP]
               [-upm UPDATE_PARAM] [-db DEBUG] [-lg_d LOG_DIR]
               [-bs BATCH_SIZE] [-ckpt CHECK_POINT] [-bwd BEAM_WIDTH]
               [-vbs VALID_BATCH_SIZE] [-evalset {test,dev}]
optional arguments:
  -h, --help            show this help message and exit
  -t TASK, --task TASK
  -g GPU, --gpu GPU
  -dw DATA_WORKER, --data_worker DATA_WORKER
  -fd FEATURE_DIM, --feature_dim FEATURE_DIM
  -corp_dir CORPUS_DIR, --corpus_dir CORPUS_DIR
  -corp_tr CORPUS_TRAIN, --corpus_train CORPUS_TRAIN
  -corp_te CORPUS_TEST, --corpus_test CORPUS_TEST
  -corp_de CORPUS_DEV, --corpus_dev CORPUS_DEV
  -vp VIDEO_PATH, --video_path VIDEO_PATH
  -op OPTIMIZER, --optimizer OPTIMIZER
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
  -wd WEIGHT_DECAY, --weight_decay WEIGHT_DECAY
  -mt MOMENTUM, --momentum MOMENTUM
  -nepoch NUM_EPOCH, --num_epoch NUM_EPOCH
  -us UPDATE_STEP, --update_step UPDATE_STEP
  -upm UPDATE_PARAM, --update_param UPDATE_PARAM
  -db DEBUG, --DEBUG DEBUG
  -lg_d LOG_DIR, --log_dir LOG_DIR
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
  -ckpt CHECK_POINT, --check_point CHECK_POINT
  -bwd BEAM_WIDTH, --beam_width BEAM_WIDTH
  -vbs VALID_BATCH_SIZE, --valid_batch_size VALID_BATCH_SIZE
  -evalset {test,dev}, --eval_set {test,dev}
```

### Train
To train on RWTH-PHOENIX-Weather-2014 with C3D features, run
```bash
python main.py --task train 
               --batch_size 20 
               --log_dir ./log/reimp 
               --learning_rate 1e-4 
               --data_worker 8 
               --video_path C3D_FEATURE_DIR 
               --gpu GPU_ID
```
Configure your own settings by modify the parameters if necessary.

### Test
Download the pretrained model weights with the following link:
- Model Weights [[Google Drive](https://drive.google.com/file/d/1f4G5pAxTngTJgGUnNCp-_J7iZ8gdcdai/view?usp=sharing)] [[Baidu Drive](https://pan.baidu.com/s/1xx3Qao5AFLizydKzp7-uRQ) (pwd: guh3)] [[Rec](https://rec.ustc.edu.cn/share/6ba26a00-b227-11ea-8e8d-052ba027f9bc)]

Evaluates the model on RWTH-PHOENIX-Weather-2014 development set (Dev) and testing set (Test),
```bash
python main.py --task test
               --batch_size 1
               --check_point MODEL_WEIGHTS_FILE
               --eval_set test
               --data_worker 8
               --video_path C3D_FEATURE_DIR
               --gou GPU_ID
```

- Reimplementation results:
<table style="float:center">
  <tr align="center">
    <th rowspan="2">DilatedSLR</th>
    <th colspan="3">Dev (%)</th>
    <th colspan="3">Test (%)</th>
  </tr>
  <tr align="center">
    <td>WER</td>
    <td>Del</td>
    <td>Ins</td>
    <td>WER</td>
    <td>Del</td>
    <td>Ins</td>
  </tr>
  <tr align="center">
    <td>w/o post-processing</td>
    <td>37.4</td>
    <td>8.6</td>
    <td>4.9</td>
    <td>37.1</td>
    <td>8.5</td>
    <td>4.3</td>
  </tr>
  <tr align="center">
      <td>+w post-processing</td>
      <td>32.2</td>
      <td>11.0</td>
      <td>4.4</td>
      <td>31.9</td>
      <td>11.0</td>
      <td>3.7</td>
    </tr>
</table>
The official evaluation script merges some sign words with similar meaning but different label, "post-processing" corresponds to the resutls with such operations.

## Related Resources
- [Visual Sign Language Research, USTC](http://home.ustc.edu.cn/~pjh/openresources/slr/index.html)
- [Chinese Sign Language Recognition Datasets](http://home.ustc.edu.cn/~pjh/openresources/cslr-dataset-2015/index.html)
- Related Paper
  * Junfu Pu, Wengang Zhou, and Houqiang Li, "Iterative Alignment Network for Continuous Sign Language Recognition," *CVPR*, 2019. [[PDF](http://openaccess.thecvf.com/content_CVPR_2019/papers/Pu_Iterative_Alignment_Network_for_Continuous_Sign_Language_Recognition_CVPR_2019_paper.pdf)]
  * Hao Zhou, Wengang Zhou, Yun Zhou, and Houqiang Li, "Spatial-Temporal Multi-Cue Network for Continuous Sign Language Recognition," *AAAI*, 2020. [[PDF](https://arxiv.org/pdf/2002.03187.pdf)]
  
## Contact
If you have any questions, please feel free to contact pjh@mail.ustc.edu.cn.
