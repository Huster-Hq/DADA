<div align="center">
<h1>Targeted False Positive Synthesis via Detector-guided Adversarial Diffusion Attacker for Robust Polyp Detection</h3>


## Overview
In this paper, we tackle the challenge of false positives in polyp detection by proposing a novel **targeted false positive synthesis method**. Unlike existing approaches that primarily focus on enhancing polyp diversity, our method emphasizes generating **high-value negative samples** to improve the quality of training data for polyp detectors. To achieve this, we design an innovative **adversarial diffusion framework** that integrates a background denoiser and a detector-guided adversarial diffusion attacker.

The proposed approach consists of two core components: First, the **Background-only Denoiser (BG-De)**, a diffusion-based generator trained to focus exclusively on learning diverse background patterns by masking polyp regions during training. Second, the **Detector-guided Adversarial Diffusion Attacker (DADA)**, which perturbs the denoising process to generate challenging false positives capable of confusing a well-trained polyp detector, thereby guiding the synthesis of high-value negative samples.

This work is the first to apply adversarial diffusion to lesion detection, establishing a new paradigm for **targeted false positive generation**. Extensive experiments on the public **Kvasir dataset** and an in-house dataset demonstrate significant improvements in F1-score for state-of-the-art detection models such as YOLO and DETR, achieving at least **2.6% and 2.7% enhancements**, respectively.

Our method provides a new perspective for building more robust and reliable polyp detection systems, effectively reducing false positives and enhancing the clinical applicability of colorectal cancer screening.

<p align="center">
    <img src="figs/overview.jpg"/> <br />
</p>
## Performance on Kvasir and In-house

### 1. Quantitative Comparisons
<p align="center">
    <img src="figs/fig3.png"/> <br />
</p>
### 2. Qualitative Comparisons
<p align="center">
    <img src="figs/fig2.jpg"/> <br />
</p>



## Quick start
#### 1. Install dependencies.
```bash
cd DADA
pip install -r requirements.txt
```

#### 2. Prepare the datasets for BG-De.
Please place the dataset you want to train in the path `./datasets` and ensure that the size of the images and masks is 256. The path structure should be as follows:
```none
  DADA
  ├── datasets
  │   ├── images
  │   ├── masks
```

#### 3.Train your own BG-De.

Please set the `"state"` parameter of modelConfig in `Main.py` to `"train"`, And set parameters such as batch_size according to actual conditions, Place the weights in the `BG-De_model` folder.

```
python Main.py
```

We provide pre-trained weights on the kvasir dataset, which can be downloaded from the following link:（这里放置预训练好的BG-De权重链接）.

### - Testing

Before running the tests, ensure an object detection model is ready. For this project, we utilize the YOLOv5l architecture. Pre-trained weights, optimized on the Kvasir dataset, can be downloaded from the following link: （这里放置目标检测预训练权重链接）,Place the weights in the `Detection_model` folder and set the `"state"` parameter of `modelConfig` in `Main.py` to `"eval"`.

```bash
python Main.py
```

## Acknowledgments
Thanks [DenoisingDiffusionProbabilityModel](https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-) for implementing an efficient DDPM and using it as the base method in this work.

