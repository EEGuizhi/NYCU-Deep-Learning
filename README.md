# NYCU Deep Learning

This repository contains laboratory assignments and implementations for the Deep Learning course offered by the Institute of Electronics at National Yang Ming Chiao Tung University (NYCU).

## 📚 Course Information
- **Course Name**: Deep Learning
- **Institution**: Institute of Electronics, National Yang Ming Chiao Tung University
- **Academic Year**: 2025

## 📋 Table of Contents
- [Overview](#overview)
- [Labs](#labs)
  - [Lab 01: Backpropagation](#lab-01-backpropagation)
  - [Lab 02: Crowd Counting](#lab-02-crowd-counting)
  - [Lab 03: Machine Translation](#lab-03-machine-translation)
- [Getting Started](#getting-started)
- [Requirements](#requirements)
- [Repository Structure](#repository-structure)
- [Resources](#resources)
- [License](#license)

## 🎯 Overview

This repository showcases practical implementations of deep learning concepts through three comprehensive laboratory assignments. Each lab focuses on different aspects of deep learning, from fundamental backpropagation to advanced computer vision and natural language processing tasks.

## 🔬 Labs

### Lab 01: Backpropagation
**Focus**: Neural Network Fundamentals and Classification

Build an accurate neural network for classification on the Fashion-MNIST dataset with the following implementations:
- **Task 1**: CNN built from scratch using NumPy (forward/backward propagation, optimizers, loss functions)
- **Task 2**: Equivalent CNN using PyTorch

**Key Techniques**:
- Data augmentation
- Center loss
- Balanced data sampling
- Adam optimizer

**Achievement**: Improved model accuracy from 90% to 94% on validation set without increasing parameters. Ranked **3rd out of 139 participants** on the private test set with 94.8% accuracy.

**[→ View Lab 01 Details](./Lab_01/README.md)**

### Lab 02: Crowd Counting
**Focus**: Computer Vision and Density Estimation

Implementation of crowd counting techniques using deep learning models.

**Key Components**:
- CDenseNet architecture
- Custom training loops
- Heatmap generation using YOLOv8

**[→ View Lab 02 Details](./Lab_02/README.md)**

### Lab 03: Machine Translation
**Focus**: Natural Language Processing and Sequence-to-Sequence Models

Implementation of machine translation from Chinese to English using deep learning models.

**Key Features**:
- Sequence-to-sequence architecture
- BLEU score evaluation
- Custom tokenizers for Chinese and English

**[→ View Lab 03 Details](./Lab_03/README.md)**

## 🚀 Getting Started

### Prerequisites
```bash
# Python 3.x required
# Install dependencies (adjust based on specific lab requirements)
pip install numpy pandas matplotlib torch torchvision
```

### Running the Labs

Each lab contains Jupyter notebooks and/or Python scripts:

```bash
# Lab 01 - NumPy Implementation
cd Lab_01
python Lab01_task1_exe.py

# Lab 03 - Machine Translation Testing
cd Lab_03
python run.py [model_path] [data_path]
```

For detailed instructions, refer to the README in each lab directory.

## 📦 Requirements

- Python 3.x
- NumPy
- PyTorch
- Pandas
- Matplotlib
- Jupyter Notebook (for .ipynb files)

Additional requirements may be specified in individual lab directories.

## 📁 Repository Structure

```
NYCU-Deep-Learning/
├── Lab_01/                      # Backpropagation Lab
│   ├── model/                   # NumPy CNN implementation
│   ├── logs/                    # Training logs
│   ├── Lab01_task1.ipynb        # NumPy implementation notebook
│   ├── Lab01_task2.ipynb        # PyTorch implementation notebook
│   ├── Lab01_task1_exe.py       # Executable Python version
│   ├── Lab01_report.pdf         # Lab report
│   └── README.md                # Lab 01 documentation
├── Lab_02/                      # Crowd Counting Lab
│   ├── training_tools/          # Training scripts
│   ├── figures/                 # Visualization outputs
│   ├── logs/                    # Training logs
│   ├── CDenseNet.py             # Model implementation
│   ├── Lab02_CDenseNet.ipynb    # Demo notebook
│   ├── Lab02_report.pdf         # Lab report
│   └── README.md                # Lab 02 documentation
├── Lab_03/                      # Machine Translation Lab
│   ├── training_tools/          # Training scripts
│   ├── logs/                    # Training logs
│   ├── network.py               # Model architecture
│   ├── utils.py                 # Utility functions
│   ├── run.py                   # Testing script
│   ├── Lab03_MachineTranslation.ipynb  # Demo notebook
│   ├── Lab03_report.pdf         # Lab report
│   └── README.md                # Lab 03 documentation
├── Colab_Tutorial_2025.pdf      # Colab setup tutorial
├── LICENSE                      # License information
└── README.md                    # This file
```

## 📖 Resources

### Datasets
- **Lab 01**: [Fashion-MNIST Dataset](https://drive.google.com/drive/folders/1CG1yxgXLU2at19rPChJ4jxg4UXy1h6sq?usp=drive_link)
- **Lab 02**: [Crowd Counting Dataset](https://drive.google.com/drive/folders/1aHAQPQySPvMWT6mtEoNKHn3gMUJ1K5cl?usp=drive_link)
- **Lab 03**: [Translation Dataset](https://drive.google.com/drive/folders/18CS4PgD5BfDbLUWnrJzEVVUzhctG3SAe?usp=drive_link)

### Pre-trained Models
- **Lab 01**: [Model Weights](https://drive.google.com/drive/folders/1z3MCduA288mZ5ZaKoLXh4mVFQodUo8xf?usp=drive_link)
- **Lab 02**: [Model Weights](https://drive.google.com/drive/folders/1a3ygru7KJgSwqUfuSCB7lstU5pJj6-zW?usp=drive_link)
- **Lab 03**: [Model Weights](https://drive.google.com/drive/folders/1Cc6TsoxM3ZDvfny2TA87j_QYtok8hHVj?usp=drive_link)

### Course Materials
- [Colab Tutorial 2025](./Colab_Tutorial_2025.pdf) - Guide for using Google Colab for the assignments

## 📄 License

This project is licensed under the terms specified in the [LICENSE](./LICENSE) file.

## 🙏 Acknowledgments

- National Yang Ming Chiao Tung University, Institute of Electronics
- Deep Learning Course Instructors and TAs
- All contributors and fellow students

---

**Note**: For detailed information about each lab, including methodology, results, and implementation details, please refer to the individual lab reports and README files in their respective directories.
