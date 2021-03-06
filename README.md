# Polarized Self-Attention: Towards High-quality Pixel-wise Regression
This is an official implementation of:

**Huajun Liu, Fuqiang Liu, Xinyi Fan and Dong Huang**. ***Polarized Self-Attention: Towards High-quality Pixel-wise Regression*** [Arxiv Version](https://arxiv.org/abs/2107.00782)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/polarized-self-attention-towards-high-quality-1/pose-estimation-on-coco-test-dev)](https://paperswithcode.com/sota/pose-estimation-on-coco-test-dev?p=polarized-self-attention-towards-high-quality-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/polarized-self-attention-towards-high-quality-1/keypoint-detection-on-coco)](https://paperswithcode.com/sota/keypoint-detection-on-coco?p=polarized-self-attention-towards-high-quality-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/polarized-self-attention-towards-high-quality-1/semantic-segmentation-on-cityscapes-val)](https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes-val?p=polarized-self-attention-towards-high-quality-1)

### Citation: 

```bash
@article{Liu2021PSA,
  title={Polarized Self-Attention: Towards High-quality Pixel-wise Regression},
  author={Huajun Liu and Fuqiang Liu and Xinyi Fan and Dong Huang},
  journal={Arxiv Pre-Print arXiv:2107.00782 },
  year={2021}
}
```

## Codes and Pre-trained models will be uploaded soon~

### Top-down 2D pose estimation models pre-trained on the MS-COCO keypoint task(Table4 in the Arxiv version).
 
| Model Name              | Backbone              |Input Size | AP | pth file |
| :----------------------: | :---------------------:| :--------------: | :--------------: | :------------:  |
| UDP-Pose-PSA(p)     | HRNet-W48         |256x192 |78.9              | to be uploaded |
| UDP-Pose-PSA(p)     | HRNet-W48         |384x288 |79.5             | to be uploaded |
| UDP-Pose-PSA(s)     | HRNet-W48         |384x288 |79.4              |to be uploaded   |

#### Setup and inference:


### Semantic segmentation models pre-trained on Cityscapes (Table5 in the Arxiv version).
 
| Model Name              | Backbone               | val mIoU | pth file |
| :----------------------: | :---------------------:| :--------------: | :------------:  |
| HRNetV2-OCR+PSA(p)     | HRNetV2-W48          |86.95              | [download](https://cmu.box.com/s/if90kw6r66q2y6c5xparflhnbwi6c2yi)  |
| HRNetV2-OCR+PSA(s)     | HRNetV2-W48          |86.72              | [download](https://cmu.box.com/s/uyzzfmkx8p2ipcznpzdtf14ng63s65sq)   |

#### Setup and inference:

