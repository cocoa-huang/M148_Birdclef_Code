# BirdCLEF 2025 Competition

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle Profile](https://img.shields.io/badge/Kaggle--Profile-hzkeric-blue)](https://www.kaggle.com/hzkeric) 

## Table of Contents 📋

* [Introduction](#introduction)
* [Competition Overview](#competition-overview)
* [Dataset](#Dataset-💾)
* [Methodology](#methodology-⚙️)
    * [Data Preprocessing](#data-preprocessing)
    * [Model Architecture](#model-architecture)
    * [Training](#training)
    * [Post-processing/Ensemble (if any)](#post-processingensemble-if-any)
* [Results](#results-📊)
    * [Validation Performance](#validation-performance)
    * [Competition Score (Public/Private LB)](#competition-score-publicprivate-lb)
* [Acknowledgments](#acknowledgments-🙏)
* [License](#license-📜)

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

## Dataset 💾

* **Dataset Name:** BirdCLEF 2025 Official Dataset
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

## Methodology ⚙️

This is the core section detailing your solution. You can either summarize the methodology here and point to specific notebooks for details, or detail it directly.

### Data Preprocessing

* **Notebook:** `notebooks/01_data_preprocessing.ipynb` (or your chosen descriptive name)
* **Summary:** Describe audio loading, resampling, feature extraction (spectrograms like Mel, STFT, CQT with parameters), data augmentation techniques, handling class imbalance, and your train/validation split strategy.

### Model Architecture

* **Notebook:** `notebooks/02_model_definition.ipynb` (or where the model is primarily defined/used)
* **Summary:** Specify base models (e.g., ResNet, EfficientNet, PANNs), pre-trained weights, modifications, input/output shapes. A textual description or a link to a cell in the notebook showing the model summary would be good.

### Training

* **Notebook:** `notebooks/03_model_training.ipynb`
* **Summary:** Detail the framework (PyTorch, TensorFlow), optimizer, loss function, batch size, epochs, learning rate schedule, and regularization techniques.

### Post-processing/Ensemble (if any)

* **Notebook:** `notebooks/04_prediction_and_submission.ipynb` (or a dedicated ensemble notebook)
* **Summary:** Explain thresholding strategies, ensembling methods if used, and how predictions for long audio files were handled.

---

## Results 📊

Present your key findings and performance. You can include screenshots of relevant plots from your notebooks or summarize the key metrics.

### Validation Performance

* Reference the notebook where validation results are calculated (e.g., `notebooks/03_model_training.ipynb` or a separate `notebooks/05_evaluation.ipynb`).
* Show your best model's performance on your local validation set using the competition metric.
* Include other relevant metrics or visualizations (e.g., confusion matrices).

### Competition Score (Public/Private LB)

* **Public Leaderboard Score:** [Your score] (Rank: [Your rank])
* **Private Leaderboard Score:** [Your score] (Rank: [Your rank])
* Link to your Kaggle submission or team profile if desired.

---
