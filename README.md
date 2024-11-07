# 🟤⚪️ Uncertainty in Skin Lesion Classification

Welcome to the Uncertainty in Skin Lesion Classification project! This repository contains the code and configurations to run experiments on skin lesion classification with uncertainty estimation.

## 📋 Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Dataset](#dataset)
- [Running Experiments](#running-experiments)
- [Configuring Experiments](#configuring-experiments)
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
   git clone https://tfs.university.innopolis.ru/tfs/ai_lab/MedicineResearch/_git/uncertainty_skin
   ```
   ```bash
    cd uncertainty_skin
   ```

2. **Install Dependencies and Activate Environment**:

    Conda will automatically create environment with python 3.12.4.

    Run the setup script to install the required dependencies:

    ```bash
    source setup.sh
    ```

    The setup.sh script will create a virtual environment, install the necessary packages, and configure the environment for the experiments.

    If you've activated virtual environment, your terminal prompt will have the name of the virtual environment in the brackets and look like this: 

    ```
    (uncertainty_skin) root@409d5492a9ad:/repo/uncertainty_skin#
    ```


    > If you've deactivated your virtual environment, you can activate it again by launching `setup.sh` again

## 📚 Dataset

You can manually download dataset from [kaggle](https://www.kaggle.com/datasets/olegopoly/isic-balanced)

or just chosee `Download Dataset` in main menu (after setup)
```
python uncertainty_skin
```


## 🚀 Running Experiments:

There are three modes for performing the experiments.

### Launch modes

1. #### Multiseed training and testing

    Perform full experiment in one run

    ```bash
    python uncertainty_skin/multi_seed.py +experiment=<experiment_name>
    ```

2. #### Just training

    Training on fixed seed: 

    ```bash
    python uncertainty_skin/main_train.py +experiment=<experiment_name>
    ```

    To train multiseed:

    ```bash
    python uncertainty_skin/main_train.py --multirun +experiment=<experiment_name> dataset.seed=[42,0,3,9,17]
    ```

    To train in the background add `nohup` to your command: 
    ```bash
    nohup python uncertainty_skin/main_train.py +experiment=<experiment_name> &
    ```

3. #### Just testing

    For testing you may choose appropriate pair of experiment and checkpoint (with Model and Loss function): 

    ```bash
    python uncertainty_skin/main_test.py +experiment=<experiment_name> model.checkpoint_path=<checkpoint_path> dataset.seed=<seed>
    ```

3. #### ⚡️ Dedicated menu for experiment launching [Experimental]

    You can use interactive menu to launch experiments! 

    ```bash
    python uncertainty_skin
    ```

Replace `<experiment_name>` with one of the following:

- tm_uanll_cnn: Experiment with Triplet Margin + UANLL loss using a CNN model.

- tm_uanll_resnet: Experiment with Triplet Margin + UANLL loss using a ResNet 50 model.

- debug_ce: Mode for testing training pipeline. Dataset reduced for 300 samples. (TM+CE loss, CNN model)

- debug_uanll: Mode for testing training pipeline. Dataset reduced for 300 samples. (TM+UANLL loss, CNN model)

For example, to run the experiment with the CNN model:

```bash
python uncertainty_skin/main_train.py +experiment=tm_uanll_cnn
```

## 🧪 Configuring Experiments:

Copy the existing config inside `uncertainty_skin/configs/experiment` 

The structure of the config: 
```
# @package _global_
defaults: 
  - _self_

offline: True  # clearml online sync turned off

model:
  name: CNN # additionally you can use any of the Timm encoder name or add your own
  loss_fun: 'TM+CE'
  checkpoint_path: '/repo/uncertainty_skin/outputs/2024-11-02/13-25-48/checkpoints/CNN_42_TM+UANLL_5e5183ea-9d19-493f-a1ae-799dac919ce8_epoch=0.ckpt'

dataset:
  img_size: 32
  bagging_size: 300 # size of the dataset

trainer:
  max_epochs: 1 # number of epochs

```

## 📊 Results
After running the experiments, the results will be saved in root directory. 

Without `--multirun` argument: 
```
---/
---/outputs/
-----------/{launch_date}
-------------------------/{launch_time}
```

With `--multirun` argument: 
```
---/
---/multirun/
-----------/{launch_date}
-------------------------/{launch_time}
```


The results include:

- Predictions: CSV files containing the true labels, predicted labels, and uncertainty estimates.

- Metrics: CSV files containing the accuracy, F1 score, and other metrics.

- Confusion Matrices: Visualizations of the confusion matrices for the test set.

- True/false labels histograms

## 🤝 Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## 📄 License
This project is licensed under the MIT License. See the LICENSE file for details.