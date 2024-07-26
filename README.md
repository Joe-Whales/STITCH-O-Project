# STITCH-O: Anomaly Detection in Drone-based Orthomosaics

## Overview

STITCH-O is a project aimed at identifying stitching artifacts in drone-based orthomosaics using anomaly detection techniques. This repository contains an adapted implementation of the UniAD (Unified Anomaly Detection) model, specifically tailored for detecting anomalies in orchard orthomosaic images.

## Table of Contents

- [STITCH-O: Anomaly Detection in Drone-based Orthomosaics](#stitch-o-anomaly-detection-in-drone-based-orthomosaics)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Background](#background)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Data Preparation](#data-preparation)
    - [Image chunking](#image-chunking)
  - [Model Architecture](#model-architecture)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Results](#results)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)

## Background

Stitching artifacts in orthomosaics can lead to inaccurate analysis in precision farming. This project aims to automate the detection of these artifacts using state-of-the-art anomaly detection techniques.

## Features

- Adapted UniAD model for orthomosaic anomaly detection
- Custom data loading pipeline for large-scale orthomosaic images
- Modified training loop to accommodate specific requirements of orthomosaic data
- Enhanced evaluation metrics tailored for stitching artifact detection

## Installation

Create a new virtual environment and install the required packages using the following commands:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

To preprocess the orthomosaic data, run the following command and run the training script:

```bash
pipeline.bat
```

Alternatively, you can run each component separately:

```bash
python .\Preprocessing\chunker.py preprocess_config.yaml
python ./Preprocessing/process_chunks.py chunks chunks_scaled
python .\Preprocessing\generate_metadata.py chunks_scaled
python .\UniAD\train_val.py --config train_config.yaml
```

## Data Preparation

[Explain the data preprocessing steps, including image chunking and dimension selection]
### Image chunking
Orthomosaic images are typically large in size, which can make training difficult. To address this, we chunk the images into smaller pieces and resize them to a fixed dimension. This allows us to train the model on smaller image patches. Since the focus of this project is simply detecting the presence of stitching artifacts, we have chosen to mask out regions that are not either stitching artifacts or normal orchard regions. This is done by creating a mask layer where out of scope regions are set to 0, case 1 anomalies are set to 1, case 2 anomalies are set to 2 and normal orchard regions have no value. 

## Model Architecture

This project uses an adapted version of the UniAD model. Key modifications include:

[List major changes made to the original UniAD architecture]

## Training

[Provide details about the training process, including any custom training loops or techniques]

## Evaluation

[Describe the evaluation metrics used (e.g., AUROC, Precision-Recall curves) and how they are implemented]

## Results
[Summarize the performance of the model on the orthomosaic dataset]
Results of training the model for 10 epochs:
|  clsname   |   mean   |
|:----------:|:--------:|
| all_case_2 | 0.936906 |
| all_case_1 | 0.987834 |
|    mean    | 0.96237  |
Classification accuracy for all_case_2: 0.9369058309037901
Classification accuracy for all_case_1: 0.9878338278931751

Due to the nature of the data and the fact that the case 1 anomalies get a lower anomaly score than normal images while case 2 anomalies get a higher anomaly score than normal images, the model requires two thresholds in order to accuractely classify the images. The thresholds are roughly 35 and below for case 1 anomalies and 60 and above for case 2 anomalies. These results were obtained using a 0.8 anomaly threshold in the preprocessing configuration file. 


## License

[Include information about the project's license]

## Acknowledgements

- Original UniAD implementation: [UniAD GitHub Repository](https://github.com/zhiyuanyou/UniAD/tree/main)
- [Any other acknowledgements or credits]