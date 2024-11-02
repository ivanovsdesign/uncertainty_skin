# 🟤⚪️ Uncertainty in Skin Lesion Classification

Welcome to the Uncertainty in Skin Lesion Classification project! This repository contains the code and configurations to run experiments on skin lesion classification with uncertainty estimation.

## 📋 Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Running Experiments](#running-experiments)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🌟 Overview

This project focuses on classifying skin lesions as benign or malignant using deep learning models. We incorporate uncertainty estimation techniques to improve the robustness and reliability of the predictions. The experiments are configured to run with different models and loss functions, such as:

- **TM+UANLL CNN**: Triplet margin loss with uncertainty-aware negative log-likelihood loss using a CNN model.
- **TM+UANLL ResNet**: Triplet margin loss with uncertainty-aware negative log-likelihood loss using a ResNet model.

## 🛠️ Setup

To set up the environment for running the experiments, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone ...
   cd uncertainty_skin
   ```

2. **Install Dependencies**:

    Conda will automatically create environment with python 3.12.4.

    Run the setup script to install the required dependencies:

    ```bash
    ./setup.sh
    ```

    The setup.sh script will create a virtual environment, install the necessary packages, and configure the environment for the experiments.

## 🚀 Running Experiments:

To run the experiments, use the following command:

```bash
python main_train.py +experiment=<experiment_name>
```
Replace <experiment_name> with one of the following:

- tm_uanll_cnn: Experiment with Triplet Margin + UANLL loss using a CNN model.

- tm_uanll_resnet: Experiment with Triplet Margin + UANLL loss using a ResNet model.

For example, to run the experiment with the CNN model:

```bash
python main_train.py +experiment=tm_uanll_cnn
```
## 📊 Results
After running the experiments, the results will be saved in the results directory. The results include:

- Predictions: CSV files containing the true labels, predicted labels, and uncertainty estimates.

- Metrics: CSV files containing the accuracy, F1 score, and other metrics.

- Confusion Matrices: Visualizations of the confusion matrices for the test set.

## 🤝 Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## 📄 License
This project is licensed under the MIT License. See the LICENSE file for details.