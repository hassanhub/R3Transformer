# R3-Transformer
This is the official code release for R3-Transformer proposed in [Neuro-Symbolic Representations for Video Captioning: A Case for Leveraging Inductive Biases for Vision and Language](https://arxiv.org/abs/2011.09530).

## Installation
### Option (I)
All dependencies are included in the original model's container.
First install the latest [docker](https://docs.docker.com/engine/install/ubuntu/). Then pull our docker image by:
```
docker pull hassanhub/vid_cap:latest
```
Then run the container by:
```
docker run --gpus all --name r3_container -it -v /home/
```

Note: This image already includes CUDA-related drivers and dependencies.

### Option (II)
Alternatively, you can create your own environment and make sure the following dependencies are installed:

- `Python 3.7/3.8`
- `Tensorflow 2.3`
- `CUDA 10.1`
- `NVIDIA Driver v 440.100`
- `CuDNN 7.6.5`
- `opencv-python`
- `h5py`
- `transformers`
- `matplotlib`
- `scikit-image`
- `nvidia-ml-py3`
- `decord`
- `pandas`
- `tensorcore.dataflow`

## Data Preparation
In order to speed-up data infeed, we utilize a multi-chunk [hdf5](https://github.com/h5py/h5py) format.
There are two options for getting data prepared for train/evaluation.
### Option (I)
Download pre-extracted features using SlowFast-50-8x8 pre-trained on Kinetics 400 from this link:
- Parts 0-10 (coming soon...)

### Option (II)
Alternatively, you can follow these steps to extract a customized version of features using your own visual backbone:
- Download [YouCook II](http://youcook2.eecs.umich.edu/download)
- Download [ActivityNet Captions](https://cs.stanford.edu/people/ranjaykrishna/densevid/)
- Pre-process raw video files using this [script](https://github.com/hassanhub/SlowFast_TF2/blob/main/pre_process_videos.py)
- Extract visual features using your visual backbone or our pre-trained SlowFast-50-8x8 using this [script](https://github.com/hassanhub/SlowFast_TF2/blob/main/extract_3d_features.py)
- Store features and captions in a multi-chunk hdf5 format using this [script](https://github.com/hassanhub/R3Transformer/blob/main/data_utils/store_data.py)
