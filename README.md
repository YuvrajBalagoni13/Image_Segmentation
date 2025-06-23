# Image Segmentation Project

## Overview
This project implements and compares several image segmentation architectures on the CamVid dataset. I developed multiple segmentation models including UNet, Attention UNet, UNet with ResNet50 backbone, and Attention UNet with ResNet50 backbone. The best-performing model was **Attention UNet with ResNet50 backbone**, achieving **95.99% accuracy**.

## Key Models

### 1. UNet Architecture
UNet is a convolutional neural network architecture for biomedical image segmentation. Its key features include:
- **Encoder-Decoder Structure**: 
  - Encoder captures context through downsampling
  - Decoder enables precise localization through upsampling
- **Skip Connections**: Direct connections between encoder and decoder layers preserve spatial information
- **Symmetric Architecture**: U-shaped design with equal depth in encoder and decoder paths

```
Input → Encoder (Downsampling) → Bottleneck → Decoder (Upsampling) → Output
```

### 2. Attention UNet
Attention UNet enhances the standard UNet architecture with attention mechanisms:
- **Attention Gates**: Learn to focus on relevant regions while suppressing irrelevant areas
- **Soft Attention**: Computes attention coefficients for each spatial location
- **Benefits**:
  - Improved feature representation in skip connections
  - Better handling of varying object sizes
  - Reduced computational load on irrelevant regions

```
Input → Encoder → Attention Gates → Bottleneck → Decoder with Attended Features → Output
```

## Model Comparison
| Model | Backbone | Accuracy | Key Features |
|-------|----------|----------|--------------|
| UNet | None | 92.5% | Standard encoder-decoder with skip connections |
| Attention UNet | None | 93.8% | Adds attention gates to skip connections |
| UNet | ResNet50 | 94.6% | Uses ResNet50 as encoder for feature extraction |
| **Attention UNet** | **ResNet50** | **95.99%** | **Combines ResNet features with attention mechanism** |

## Results
The Attention UNet with ResNet50 backbone achieved the best performance on the CamVid dataset:

### Key Metrics
- **Accuracy**: 95.99%
- **Dice Coefficient**: 0.85
- **mIoU**: 0.82
- **Loss**: 0.12

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- Torchvision
- OpenCV
- Matplotlib

### Installation
```bash
git clone https://github.com/YuvrajBalagoni13/Image_Segmentation.git
cd Image_Segmentation
pip install -r requirements.txt
```

### Training
```bash
python main.py --model attention_resnet_unet --dataset camvid --epochs 50
```

## File Structure
```
Image_Segmentation/
├── Models/             
│   ├── UNet.py
│   ├── AttentionUNet.py
│   ├── R50UNet.py
│   └── R50AttentionUNet.py
├── Dataset/                
│   └── CamVidDataset.py
├── Metrics/               
│   └── metrics.py
├── Utils/               
│   └── plots.py
|── Engine/               
│   |── train.py
│   └── inference.py     
├── main.py            
└── requirements.txt     
```

## References
1. Ronneberger O., et al. (2015). [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
2. Oktay O., et al. (2018). [Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/abs/1804.03999)
3. Brostow G. J., et al. (2008). [Semantic Object Classes in Video: A High-Definition Ground Truth Database](https://link.springer.com/article/10.1007/s11263-009-0255-8) (CamVid Dataset)
4. He K., et al. (2016). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (ResNet)
