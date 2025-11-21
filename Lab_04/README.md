# Lab 04: Model Compression:
### **Description**
(pass)

### **Dataset**
We use CIFAR-10 dataset in this lab.
To download the dataset, please see:</br>
`task1_pruning.ipynb`, `task2_1_quantization_api.ipynb`, `task2_2_quantization_manual.ipynb`

### **Trained Model Weights**
To download the trained model weights, please see:</br>
https://drive.google.com/drive/folders/1L7000fWBZoBDU3C8tPvY6x1--uoU54JM?usp=drive_link

### **Performance Ranking**
- **Task 1 Score**:  9/10
- **Task 2 Score**: 10/10

### **Folder Structure**
```
Lab_04/
├── figures/
│   └── (Some figures showed in the report)
├── logs/
│   ├── (Task 1 training log .csv file)
│   └── (Task 1 pruning info .txt files)
├── tools/
│   └── (Tools used in task 1)
├── 2025_DL_Lab04.pdf                 (Assignment description)
├── task1_pruning.ipynb               (Demo file)
├── task2_1_quantization_api.ipynb    (Demo file)
├── task2_2_quantization_manual.ipynb (Demo file)
├── resnet20.py                       (The base model used in this lab)
├── resnet20_int8.py                  (The quantized model used in task 2-2)
├── check_pruning.py                  (Evaluation tool for task 1)
├── check_quantization.py             (Evaluation tool for task 2-2)
└── Lab04_report.pdf                  (Lab report)
```
