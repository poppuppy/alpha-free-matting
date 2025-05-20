<h1> Alpha-Free Matting</h1>
<h3> Training Matting Models without Alpha Labels</h3>

Wenze Liu<sup>1</sup>, Zixuan Ye<sup>2</sup>, Hao Lu<sup>2</sup>, Zhiguo Cao<sup>2</sup>, Xiangyu Yue<sup>1</sup>

<sup>1</sup> CUHK, <sup>2</sup> HUST
[![arXiv](https://img.shields.io/badge/arXiv%20paper-2408.10539-b31b1b)](https://arxiv.org/abs/2408.10539)

## Abstract

The labeling difficulty has been a longstanding problem in deep image matting. To escape from fine labels, this work explores using rough annotations such as trimaps coarsely indicating the foreground/background as supervision. We present that the cooperation between learned semantics from indicated known regions and proper assumed matting rules can help infer alpha values at transition areas. Inspired by the nonlocal principle in traditional image matting, we build a directional distance consistency loss (DDC loss) at each pixel neighborhood to constrain the alpha values conditioned on the input image. DDC loss forces the distance of similar pairs on the alpha matte and on its corresponding image to be consistent. In this way, the alpha values can be propagated from learned known regions to unknown transition areas. With only images and trimaps, a matting model can be trained under the supervision of a known loss and the proposed DDC loss. Experiments on AM-2K and P3M-10K dataset show that our paradigm achieves comparable performance with the fine-label-supervised baseline, while sometimes offers even more satisfying results than human-labeled ground truth.

## Get Started

* Install [ViTMatte](https://github.com/hustvl/ViTMatte). (Warning: different from VitMatte, the input (image only, without trimap) to our model contains 3 channels. Therefore the preprocess.py is slightly different.)
* Prepare the dataset, e.g., [AM-2K](https://github.com/JizhiziLi/GFM), and generate the trimaps using

```
python alpha2trimap.py --alpha_path /your/alpha/path --trimap_path /your/trimap/path
```
* Specify the image and trimap path in configs/common/dataloader.py.

* Run 
```
bash train.sh
```
to train the model.

## Citation
```
@article{liu2024training,
  title={Training Matting Models without Alpha Labels},
  author={Liu, Wenze and Ye, Zixuan and Lu, Hao and Cao, Zhiguo and Yue, Xiangyu},
  journal={arXiv preprint arXiv:2408.10539},
  year={2024}
}
```
