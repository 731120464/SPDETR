# SP-DETR:Superior-Point Weak Semi-supervised DETR for Crop and Weed Detection

## Introduce

* Introduces a Point Amplification Module (PAM) for many-to-one mapping between points and objects.

* Proposes a Multi-branch Point Encoder (MPE) with Relative Position Embedding (RPE) and Semantic Alignment Encoder (SAE).

* Designs a Denoised Decoder (DD) for stable Hungarian matching and faster convergence.

* Incorporates a Colour-guided Point Annotation (CPA) strategy for precise crop/weed annotation.

* Achieves state-of-the-art performance on the Bonn Weed Detection (BWD) dataset.

## Requirements

```
ubuntu 20.04
python 3.6.13
GPU NVIDIA GeForce GTX1080Ti 11G
```

## Installation

```
git clone https://github.com/731120464/SPDETR.git
cd SPDETR
conda create -n spdetr python=3.6
conda activate spdetr
pip install -r requirements.txt
```


## Data Preparation

* The BWD dataset ... http://www.ipb.uni-bonn.de/data/sugarbeets2016/
* 20% image ids ```in ./datasets/annoted_img_ids.py && ./cvpods/datasets/annoted_img_ids.py```
* pretrained [baseline-checkpoint0107.pth](https://pan.baidu.com/s/1oiUDZqCk5D8bQjmMqb5ydw?pwd=3pux) at 20%.


#### 1. Train SPDETR by 20% bbox

* ```python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path ./datasets/bwd  --partial_training_data --output_dir ./ckpt-ps/point-detr-9x --epochs 108 --lr_drop 72 --data_augment --position_embedding sine --warm_up --multi_step_lr```

#### 2. Generate 80% pseudo-bbox 

* ```python3 main.py --coco_path ./datasets/bwd --generate_pseudo_bbox --generated_anno SPDETR --position_embedding sine --resume ./ckpt-ps/baseline-checkpoint0107.pth```

-------  Student Model -------

#### Install [cvpods](https://github.com/Megvii-BaseDetection/cvpods)

#### 3. Train the student model with 20% bbox + 80% pseudo-bbox

* ```cd ./cvpods/playground/detection/coco/fcos-20p-pointdetr```
* ``` pods_train --num-gpus 8 --dir . ```

#### 4. (optional) Train the student model with 20% bbox only. 

* ```cd ./cvpods/playground/detection/coco/fcos-20p-no_teacher```
* ``` pods_train --num-gpus 8 --dir . ```

## Citation

If this work helps your research / work, please consider citing:
 ```
@article{xu2024spdetrr,
  title={SP-DETR: Superior Point Weak Semi-supervised DETR for Crop and Weed Detection},
  author={Xu, Yifei and Ren, Shuaiqiang and Li, Li and Wei, Pingping and Deng, Hao and Wang, Aichen and Rao, Yuan},
  journal={IEEE Transactions on Multimedia},
  year={2024},
  doi={10.1007/s11263-024-02005-x}
}
 ```
## Aknowledgement

This project is built upon DETR, Point-DETR, and related works in weak/semi-supervised object detection.

