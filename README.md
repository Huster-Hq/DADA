<div align="center">
<h1>Targeted False Positive Synthesis via Detector-guided Adversarial Diffusion Attacker for Robust Polyp Detection</h1>
</div>


## Overview
To tackle the challenge of polyp detectors easy to produce false positive prediction, this work firstly (to the best of our knowledge) to apply adversarial diffusion framwork to generate ***high-value negative samples*** (e.g., polyp-like interferences, light spots, circular lumens and and other background objects).
The core module is ***Detector-guided Adversarial Diffusion Attacker (DADA)***, which perturbs the denoising process to generate challenging false positives capable of confusing a well-trained polyp detector, thereby guiding the synthesis of high-value negative samples.
The negative samples synthesized by our method can improve the performance of the detector more than other data data synthesis methods.

<p align="center">
    <img src="figs/overview.jpg"/ width=800> <br />
</p>


## Experimental Results
### 1. Comparison with State-of-the-arts
### - Quantitative Comparisons
<p align="center">
    <img src="figs/fig3.png"/ width=800> <br />
</p>

### - Qualitative Comparisons
<p align="center">
    <img src="figs/fig2.jpg"/ width=800> <br />
</p>


## Quick Start

### - Preliminaries

- Python 3.8+
- PyTorch 2.2+ 
- TorchVision corresponding to the PyTorch version
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-11-7-0-download-archive)

#### 1. Install dependencies.
```bash
cd DADA
pip install -r requirements.txt
```

#### 2. Data synthesis

We provide well-trained BG-De weights based on the public Kvasir dataset, which can be downloaded from this [link](), . Please place the weights in the `BG-De_model` folder.

Additionally, we also offer YOLOv5l weights trained on the public Kvasir dataset, which can be downloaded from this [link](). Please place the weights in the `Detection_model` folder.

```
python Main.py
```




## Training
#### 2. Prepare the datasets for BG-De.
Please place the dataset you want to train in the path `./datasets` and ensure that the size of the images and masks is 256. The path structure should be as follows:
```
  DADA
  ├── datasets
  │   ├── images
  │   ├── masks
```

#### 3.Train your own BG-De.
Please set the `"state"` parameter of modelConfig in `Main.py` to `"train"`, and set parameters such as batch_size according to actual conditions.
```
python Main.py
```
Place the weights in the `BG-De_model` folder.


### - Testing
Before running the tests, ensure an object detection model is ready. For this project, we utilize the YOLOv5l architecture. Place the weights in the `Detection_model` folder and set the `"state"` parameter of `modelConfig` in `Main.py` to `"eval"`.

```bash
python Main.py
```

## Acknowledgments
We acknowledge the outstanding open-source contributions from [DDPM](https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-), [YOLOv5](https://github.com/ultralytics/yolov5), and [DETR](https://github.com/facebookresearch/detr).

