# R3Transformer
This is the official code release for R3-Transformer proposed in [Neuro-Symbolic Representations for Video Captioning: A Case for Leveraging Inductive Biases for Vision and Language](https://arxiv.org/abs/2011.09530).

## Installing
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
- `CUDA 10`
- `NVIDIA Driver`
