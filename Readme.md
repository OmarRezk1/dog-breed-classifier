# Stanford Dogs Breed Classification with PyTorch

A custom 4-block Convolutional Neural Network (CNN) built entirely from scratch using PyTorch to tackle fine-grained image classification across the 120 complex dog breeds in the Stanford Dogs Dataset.

---

## 📌 Project Overview

Fine-grained image classification (like distinguishing between highly similar dog breeds) requires robust feature extraction and careful regularization to prevent overfitting. This repository features a modular, end-to-end deep learning pipeline designed for custom PyTorch model training, state dictionary management, and evaluation.

### Key Features
* **Custom 4-Block CNN Architecture:** Designed from scratch with progressive channel expansion, batch normalization, and spatial dropout.
* **Fine-Grained Recognition:** Trained to separate 120 visually subtle classes from the Stanford Dogs dataset.
* **Modular Pipeline:** Clean separation of concerns across dataset loading, model definitions, training routines, and validation scripts.
* **State Dict Management:** Safe checkpointing and reloading of PyTorch model weights (`.pth`).

---

## 🏗️ Model Architecture

The network processes RGB images through four sequential feature extraction blocks followed by a dense classification head:

```text
Input (3 x 224 x 224)
 │
 ├── Block 1: Conv2d (3 → 64, 3x3) ➔ BatchNorm2d ➔ ReLU ➔ MaxPool2d (2x2)
 ├── Block 2: Conv2d (64 → 128, 3x3) ➔ BatchNorm2d ➔ ReLU ➔ MaxPool2d (2x2)
 ├── Block 3: Conv2d (128 → 256, 3x3) ➔ BatchNorm2d ➔ ReLU ➔ MaxPool2d (2x2)
 └── Block 4: Conv2d (256 → 512, 3x3) ➔ BatchNorm2d ➔ ReLU ➔ MaxPool2d (2x2)
 │
 └── Classifier Head: Flatten ➔ Linear (512 * H * W → 512) ➔ ReLU ➔ Dropout (p=0.5) ➔ Linear (512 → 120 Classes)