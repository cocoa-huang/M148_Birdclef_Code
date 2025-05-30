# BirdCLEF 2025 Competition

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle Profile](https://img.shields.io/badge/Kaggle--Profile-hzkeric-blue)](https://www.kaggle.com/hzkeric) 

## Table of Contents 📋

* [Introduction](#introduction)
* [Competition Overview](#competition-overview)
* [Dataset](#dataset)
* [Methodology](#methodology)
    * [Data Preprocessing](#data-preprocessing)
    * [Model Architecture](#model-architecture-&-training)
    * [Training](#training)
* [Inference/Submission](#inference-submission)
* [Results](#results)
    * [Competition Score (Public/Private LB)](#competition-score-publicprivate-lb)

---

## Introduction

This repository documents our project for the **BirdCLEF 2025 competition**. The primary goal is to identify bird species from continuous audio recordings, a challenging multi-label classification task. Our entire workflow, from data exploration and preprocessing to model training and submission generation, is presented through a series of Jupyter Notebooks for clarity and reproducibility. We aim to develop an efficient model for diverse acoustic environments.

---

## Competition Overview

* **Competition Name:** BirdCLEF 2025
* **Organizer:** LifeCLEF in partnership with the Cornell Lab of Ornithology, Macaulay Library, and xeno-canto.
* **Objective:** To develop a system that can process continuous audio recordings from various locations and automatically provide a multi-label list of bird species present in each recording.
* **Evaluation Metric:** macro-averaged ROC-AUC
* **Link to Competition:** [https://www.kaggle.com/competitions/birdclef-2025/overview](https://www.kaggle.com/competitions/birdclef-2025/overview)

---

## Dataset

* **Dataset Name:** [BirdCLEF 2025 Official Dataset](https://www.kaggle.com/competitions/birdclef-2025/data)
* **Source:** Data is primarily sourced from [xeno-canto.org](https://xeno-canto.org/) and the [Macaulay Library at the Cornell Lab of Ornithology](https://www.macaulaylibrary.org/).
* **Description:**
    * The training dataset (`train.csv`) comprises 28,564 entries and 13 columns.
    * The associated taxonomy data (`taxonomy.csv`) includes 206 unique bird species across 5 columns.
    * The dataset is characterized by minimal missing data, with missing values occurring only in geographical coordinates (latitude and longitude, 809 missing values each).
    * No duplicate records were found in the training data.
    * Recordings originate from diverse geographical locations and varied environments, resulting in a wide range of acoustic conditions and background noise levels. The EDA revealed a broad geographic spread, though some geographical outliers (514 identified via IQR) exist.
    * Recordings can contain multiple bird species (multi-label), as indicated by the `secondary_labels` column.
    * Audio quality ratings (from 0 to 5) are generally consistent, with no extreme outliers detected via the IQR method.
    * A significant finding from the EDA is the pronounced imbalance in both species counts (some species are much more frequent than others) and collection sources (the "XC" collection is dominant). This may require special handling during model development.
    * The dataset includes different types of audio files:
        * `train_audio`: 28,564 individual recordings.
        * `train_soundscapes`: 9,726 continuous soundscape recordings.
        * `test_soundscapes`: 1 continuous soundscape recording (as per the EDA file provided). *(Note: The final test set for the competition will likely be larger and released later by the organizers).*
    * The primary task involves processing these audio recordings (especially the test soundscapes) to automatically provide a multi-label list of bird species present.
    * Merging training data with taxonomy data confirmed complete species classification for all entries in `train.csv`.
* **Data Access:** Direct access on Kaggle notebook/competition.
  
---

## Methodology

This is the core section detailing your solution. You can either summarize the methodology here and point to specific notebooks for details, or detail it directly.

### Data Preprocessing

* **Exploratory Data Analysis (EDA):**
    * `01_EDA.ipynb` - Initial analysis to understand dataset structure, characteristics (species diversity, imbalance, noise, audio properties), and inform subsequent processing and modeling decisions.
* **Voice Activity Detection (VAD) - Human Voice Removal:**
    * `02_data_prep_train_audio.ipynb` - Human speech removal from `train_audio` files using the SileroVAD model. This was applied to the ~21,000 `train_audio` files containing labeled bird species.
    * `03_data_prep_train_soundscapes.ipynb` - Human speech removal from all `train_soundscapes` files using the SileroVAD model.
* **Audio Segmentation & Mel Spectrogram Generation:**
    * `04_data_prep_train_audio_mel_spec.ipynb` - Conversion of processed `train_audio` files into mel spectrograms. This implicitly follows segmentation into 5-second chunks.
    * `05_data_prep_segment_melspec_train_soundscapes.ipynb` - Segmentation of processed `train_soundscapes` into 5-second chunks and their conversion into mel spectrograms.
    * **Spectrogram Parameters:** Sample Rate: 32,000 Hz, FFT Window Size (n\_fft): 1024, Hop Length: 512, Mel Bins: 128, Window Function: Hann. Final Output Shape: Resized to (1, 256, 256).

### Model Architecture & Training

* **Initial EfficientNet_b0 model:** `06_train_initial_model_efficientnet_b0.ipynb` - Training a baseline model on `train_audio` mel spectrogram, used in later pseudolabel generation
* **ResNet model training:** `06_train_initial_CNN_model.py` - Pseudolabel generation using baseline model
* **Pseudolabel Generation:** `07_pseudo_label_generation.ipynb` - Pseudolabel generation using baseline model
* **Final Model Training with pseudolabeled data plus labeled data:** `08_train_pseudo_labeled_model_efficientenet_b0.ipynb` - Final model training utilizing the combined dataset (pseudolabel + original label)
  
---
### Inference/Submission

* **Notebook:** `inference_submissions.ipynb` 
* **Summary:** Notebook that load saved model and make competition submission

---

## Results

### Competition Score (Public/Private LB)

* **Public Leaderboard Score:** **0.787**

---
