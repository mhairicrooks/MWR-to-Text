# MWR-to-Text
# MWR-to-Text: Explainable AI for Breast Cancer Diagnostics

This repository contains the implementation of **MWR-to-Text**, the first explainable AI system that integrates Medical Microwave Radiometry (MWR) breast cancer classification with automated clinical text generation. The system uses a novel multitask learning architecture to provide both accurate diagnostic predictions and human-readable clinical explanations.

## About

Breast cancer remains a leading cause of mortality worldwide, with early detection crucial for survival. While AI-driven MWR systems show promise for radiation-free screening, they lack interpretability, hindering clinical adoption. This work introduces:

- **Multitask Learning Framework**: Combines binary classification and clinical text generation
- **Feature Projection Strategies**: Systematic comparison of prefix, concatenation, and attention-based feature injection
- **Dynamic Loss Weighting**: Automatic task balancing for stable joint optimization
- **Mode Collapse Resolution**: Sampling-based inference ensuring clinically appropriate text diversity

## Key Results

- **Classification**: 79.03% accuracy, 82.58% sensitivity, 67.92% specificity
- **Text Generation**: 0.4054 METEOR score with contextually appropriate clinical descriptions
- **Dataset Scale**: 24,000 MWR samples (significantly expanding on previous 5,000-sample studies)

## Repository Structure

**Core Implementation (`src/`)**
- `initial_model.py` - Baseline multitask model
- `model_projections.py` - Feature projection experiments  
- `data_loader.py` - MWR data preprocessing
- `classification.py` - Classification training
- `generation.py` - Text generation training

**Research Process (`notebooks/`)**
- `EDA.ipynb` - Exploratory data analysis
- `Classification.ipynb` - Classification experiments
- `Generation.ipynb` - Text generation experiments
- `Concat_Projections.ipynb` - Concatenation projection tests

**Other Files**
- `requirements.txt` - Dependencies
- `README.md` - This file

## Quick Start

### Installation

```bash
git clone https://github.com/mhairicrooks/Mwr-to-Text.git
cd Mwr-to-Text
pip install -r requirements.txt
