# Lab 01: Backpropagation
### **Description**
In this lab, the focus is on building an accurate neural network for classification on the Fashion-MNIST dataset. I implemented a simple CNN enhanced with data augmentation, center loss, balanced data sampling, and the Adam optimizer. With these techniques, I improved the model’s accuracy from 90% to 94% without increasing the number of parameters.

- In Task 1, I built a CNN from scratch using NumPy, including forward propagation, backward propagation, optimizers, and loss functions.

- In Task 2, I implemented an equivalent CNN and training loop using PyTorch.

### **Dataset**
The Fashion-MNIST dataset was shuffled and split into 50,000 + 10,000 images for the training and validation sets, and 10,000 images for the test set. In this lab, we were instructed not to modify the number of images in the training and validation sets. The final results were evaluated on a private test set, where I achieved 3rd place out of 139 participants.

To download the dataset (without the ground truth labels for the test set), please see:</br>
https://drive.google.com/drive/folders/1CG1yxgXLU2at19rPChJ4jxg4UXy1h6sq?usp=drive_link

### **Trained Model Weights**
To download the trained model weights, please see:</br>
https://drive.google.com/drive/folders/1z3MCduA288mZ5ZaKoLXh4mVFQodUo8xf?usp=drive_link

### **Folder Structure**
```
Lab_01/
├── logs/
│   └── (Training log .csv files)
├── model/
│   ├── layer.py    (Numpy implemented modules of layers in CNN)
│   └── network.py  (Numpy implemented CNN)
├── 2025_DL_Lab01.pdf   (Assignment description)
├── Lab01_task1_exe.py  (Equivalent to Lab01_task1.ipynb)
├── Lab01_task1.ipynb   (Numpy implementation)
├── Lab01_task2.ipynb   (PyTorch implementation)
└── Lab01_report.pdf    (Lab report)
```

