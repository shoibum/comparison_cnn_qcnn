--------------------------------------------------------------------------------
-- File: README.md -----------------------------------------------------------
--------------------------------------------------------------------------------

# Classical CNN vs Hybrid QCNN on PathMNIST

## Project Overview

This project implements and compares a standard Convolutional Neural Network (CNN) with a hybrid Quantum Convolutional Neural Network (QCNN) for image classification. Both models are trained and evaluated on the PathMNIST dataset, part of the MedMNIST collection.

The primary goal is to demonstrate how quantum circuits can be integrated into deep learning workflows using PyTorch and PennyLane, and to provide a basic performance comparison between the classical and hybrid quantum approaches on a medical imaging task.

**Models:**
* **Classical CNN:** A standard CNN architecture with convolutional, ReLU, pooling, and linear layers.
* **Hybrid QCNN:** Uses initial classical convolutional layers for feature extraction and dimensionality reduction, followed by a PennyLane quantum circuit (acting as a quantum processing layer), and final classical linear layers for classification.

**Dataset:**
* **PathMNIST:** A dataset of pathology images from the MedMNIST v2 collection. (3x28x28 images, 9 classes).

## Features

* Implementation of a standard CNN in PyTorch.
* Implementation of a Hybrid QCNN using PyTorch and PennyLane.
* Data loading and preprocessing for the PathMNIST dataset using `medmnist`.
* Training and evaluation loops for both models.
* Comparison of test accuracy and parameter counts.
* Visualization of validation accuracy during training.

## Requirements

* Python 3.8+
* Required libraries are listed in `requirements.txt`. Key dependencies include:
    * `torch` & `torchvision`
    * `pennylane`
    * `medmnist`
    * `numpy`
    * `matplotlib`
* (Optional) `pennylane-lightning` for potentially faster quantum simulations (requires a C++ compiler).

## Setup Instructions

1.  **Clone or Download:** Get the project files onto your local machine. Create a directory for the project. Save the Python script as `main.py`, the requirements list as `requirements.txt`, and this README as `README.md` within that directory.

2.  **Navigate to Directory:** Open your terminal or command prompt and change to the project directory.
    ```bash
    cd path/to/project/directory
    ```

3.  **(Recommended) Create and Activate Virtual Environment:**
    ```bash
    # Create environment (e.g., 'qenv')
    python -m venv qenv

    # Activate environment
    # Windows:
    qenv\Scripts\activate
    # macOS/Linux:
    source qenv/bin/activate
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

Execute the main script from your terminal within the activated virtual environment:

```bash
python main.py
```
