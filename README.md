# STITCH-O: Anomaly Detection in Drone-based Orthomosaics

## Overview

STITCH-O (Using anomaly detection to identify stitching artefacts in drone-based orthomosaics) is an advanced project aimed at automating the detection of stitching artifacts in drone-based orthomosaic images used in precision agriculture. This repository contains an adapted implementation of the UniAD (Unified Anomaly Detection) model, specifically tailored for detecting anomalies in orchard orthomosaic images.

## Table of Contents

- [STITCH-O: Anomaly Detection in Drone-based Orthomosaics](#stitch-o-anomaly-detection-in-drone-based-orthomosaics)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Background](#background)
  - [Features](#features)
  - [Disclaimer](#disclaimer)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Training](#training)
    - [Inference](#inference)
  - [Data Preparation](#data-preparation)
  - [Model Architecture](#model-architecture)
  - [Training](#training-1)
  - [Evaluation](#evaluation)
  - [Results](#results)
  - [Baseline Model](#baseline-model)
    - [Architecture](#architecture)
    - [Key Features](#key-features)
    - [Usage](#usage-1)
  - [Visualization](#visualization)
  - [Acknowledgements](#acknowledgements)

## Background

Precision agriculture increasingly relies on drone-based imaging for monitoring crop health and general farm management. These images are merged into large-scale orthomosaics, which can sometimes contain stitching artifacts that compromise data quality. Currently, these artifacts are detected through manual inspection, which is time-consuming and expensive. STITCH-O aims to automate this process using state-of-the-art anomaly detection techniques.

## Features

- Adapted UniAD model for orthomosaic anomaly detection
- Custom data loading and preprocessing pipeline for large-scale orthomosaic images
- Enhanced evaluation metrics tailored for stitching artifact detection
- Inference pipeline for whole-orchard classification
- Baseline model implementation for performance comparison

## Disclaimer
The data used in this project has not been made publicly available.

## Installation

Create a new virtual environment and install the required packages:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Training

To run the entire training pipeline (preprocessing, training, and evaluation):

```bash
train_pipeline.bat
```

Alternatively, you can run each component separately:

```bash
python .\Preprocessing\chunker.py preprocess_config.yaml
python ./Preprocessing/process_chunks.py chunks chunks_scaled
python .\Preprocessing\generate_metadata.py chunks_scaled -t
python .\UniAD\train_val.py --config train_config.yaml
```

### Inference

For inference on new data:

```bash
inference_pipeline.bat
```

This will run the following steps:
1. Segmentation of new orchard images
2. Preprocessing of segmented images
3. Running inference using the trained model
4. Classifying whole orchards based on anomaly scores

You can also run the inference script directly:

```bash
python .\UniAD\run_inference.py --config inference_config.yaml
```

## Data Preparation

The data preparation process includes:

1. Image chunking: Large orthomosaic images are divided into smaller, manageable chunks.
2. Data scaling: Pixel values are normalized to handle variations across different orchard images.
3. Train-Test split: The dataset is divided into training and testing sets.
4. Metadata generation: Structured information about the dataset is created for efficient data handling.

The preprocessing pipeline supports multiple image layers (RGB, DEM, NDVI) and can be configured using YAML files.

## Model Architecture

STITCH-O uses an adapted version of the UniAD model. Key components include:

- Feature extractor: EfficientNet B4 or ResNet50 (configurable)
- Reconstruction model: Transformer-based encoder-decoder architecture
- Custom data loading and augmentation techniques

## Training

The training process includes:

- Mixed precision training using PyTorch's GradScaler
- Customizable learning rate scheduling
- Periodic validation and model checkpointing
- Logging of training metrics using TensorBoard

Training configuration can be adjusted using the `train_config.yaml` file.

## Evaluation

Evaluation metrics include:

- Area Under the Receiver Operating Characteristic (AUROC) curve
- Classification accuracy for Case 1 and Case 2 anomalies
- Custom thresholding technique to handle both types of anomalies

The evaluation process also includes an inference pipeline for whole-orchard classification.

## Results

After training for 250 epochs:

| Anomaly Type | AUROC   |
|--------------|---------|
| Case 1       | 0.99420 |
| Case 2       | 0.96136 |
| Mean         | 0.97778 |

The model uses dual thresholding:
- Case 1 anomalies: Below ~35 (lower anomaly scores than normal images)
- Case 2 anomalies: Above ~60 (higher anomaly scores than normal images)

The STITCH-O implementation significantly outperforms the baseline model, especially for Case 2 anomalies.

## Baseline Model

The project includes a baseline model for comparison with the UniAD implementation. This baseline model serves as a benchmark to evaluate the performance improvements achieved by the more complex UniAD approach.

### Architecture

The baseline model consists of two main components:

1. **Feature Extractor**: Uses EfficientNet B4 pre-trained on ImageNet. The first four layers are used and frozen during training.

2. **U-Net Reconstruction Model**: A custom U-Net architecture designed to reconstruct the extracted features.

### Key Features

- Implements a simplified anomaly detection approach based on feature reconstruction.
- Uses Mixed Precision Training with PyTorch's GradScaler for efficient computation.
- Includes customizable learning rate scheduling options.
- Provides flexible configuration through YAML files.

### Usage

To train and evaluate the baseline model:

```bash
python baseline_model.py baseline_config.yaml
```
Performance
The baseline model achieves the following results:

Case 1 AUROC: 0.9963
Case 2 AUROC: 0.8874
Mean AUROC: 0.9419

While the baseline performs well, especially for Case 1 anomalies, it is outperformed by the UniAD implementation, particularly for the more subtle Case 2 anomalies.


## Visualization

To visualize the results and compare different experiments, use the `plot-experiments.py` script:

```bash
python plot-experiments.py /path/to/experiment/directory
```

This script generates the following plots:
1. Overall comparison of Case 1 and Case 2 AUROC across all models
2. Individual experiment results showing Case 1 AUROC, Case 2 AUROC, and Average AUROC for different model configurations

The script processes CSV files in the specified directory and its subdirectories, allowing for easy comparison of multiple experiments and model configurations.


## Acknowledgements

- Original UniAD implementation: [UniAD GitHub Repository](https://github.com/zhiyuanyou/UniAD)
- Segment Anything Model (SAM) for mask generation: [SAM GitHub Repository](https://github.com/facebookresearch/segment-anything)
- EfficientNet implementation: [EfficientNet PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)
- Project supervisor: Patrick Marais, University of Cape Town
- This project was developed as part of a CS Honours Project at the University of Cape Town

For detailed implementation and usage instructions, please refer to the individual script files and configuration YAML files in the repository.
