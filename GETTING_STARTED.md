## Getting Started with MinVIS

This document provides a brief intro of the usage of MinVIS.

Please see [Getting Started with Detectron2](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md) for full usage.


### Training & Evaluation in Command Line

We provide a script `train_net_video.py`, that is made to train all the configs provided in MinVIS.

To train a model with "train_net_video.py", first setup the corresponding datasets following
[datasets/README.md](./datasets/README.md), then download the COCO pre-trained instance segmentation weights ([R50](https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_R50_bs16_50ep/model_final_3c8ec9.pkl), [Swin-L](https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_e5f453.pkl)) and put them in the current working directory.
Once these are set up, run:
```
python train_net_video.py --num-gpus 8 \
  --config-file configs/youtubevis_2019/video_maskformer2_R50_bs32_8ep_frame.yaml
```

If the COCO pre-trained weights are in other locations, then add `MODEL.WEIGHTS /path/to/pretrained_weights` at the end to point to their locations. In addition, the configs are made for 8-GPU training for ResNet-50 and 16-GPU training for Swin-L.
Since we use ADAMW optimizer, it is not clear how to scale learning rate with batch size.
To train on 1 GPU, you need to figure out learning rate and batch size by yourself:
```
python train_net_video.py \
  --config-file configs/youtubevis_2019/video_maskformer2_R50_bs32_8ep_frame.yaml \
  --num-gpus 1 SOLVER.IMS_PER_BATCH SET_TO_SOME_REASONABLE_VALUE SOLVER.BASE_LR SET_TO_SOME_REASONABLE_VALUE
```

To evaluate a model's performance, use
```
python train_net_video.py \
  --config-file configs/youtubevis_2019/video_maskformer2_R50_bs32_8ep_frame.yaml \
  --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```
For more options, see `python train_net_video.py -h`.


### Inference Visualization with Trained Models

1. Pick a trained model and its config file. To start, you can pick from
  [model zoo](MODEL_ZOO.md),
  for example, `configs/youtubevis_2019/video_maskformer2_R50_bs32_8ep_frame.yaml`.
2. We provide `demo.py` to visualize outputs of a trained model. Run it with:
```
cd demo_video/
python demo.py --config-file ../configs/youtubevis_2019/video_maskformer2_R50_bs32_8ep_frame.yaml \
  --input /path/to/video/frames \
  --output /output/folder \  
  [--other-options]
  --opts MODEL.WEIGHTS /path/to/checkpoint_file
```
The configs are made for training, therefore we need to specify `MODEL.WEIGHTS` to a model for evaluation. The input is a folder containing video frames saved as images. For example, `ytvis_2019/valid/JPEGImages/00f88c4f0a`.

