# K-Fold Cross Validation Benchmarking for LeNet and Upgraded LeNet Models

This repository contains a Python script for running k-fold cross validation on two variations of the LeNet model: the standard LeNet-5 and an upgraded version. The script benchmarks these models by training them on the MNIST dataset, and finally plotting the mean accuracy of the models against various k values.

## Requirements

The scripts are written in Python and require the following packages:

* `torch`
* `numpy`
* `torchvision`
* `matplotlib`
* `sklearn`

Install the required packages by running the following command:
   ```
   pip install -r requirements.txt
   ```
## Setting up the conda environment 
Using the conda config file:
```
conda env create -f torch-conda-nightly.yml -n torch
```
To use this conda environment:
```
conda activate torch
```

## Running the benchmark

To run the benchmark, simply execute `main.py` in your terminal:

```
python main.py
```

This will train both models using the k-fold cross validation method. The k values used are `[2, 5, 10]`.

## Understanding the Code

Here is a high-level overview of what the main script (`main.py`) does:

1. Load the MNIST dataset and normalize it.
2. For each k value in the list, the script performs the following steps:
    1. Shuffle the dataset and split it into `k` groups or folds.
    2. For each fold, it does the following:
        1. Initialize both models and define the loss function and the optimizers.
        2. Train both models for 10 epochs using the training data for the current fold.
        3. Evaluate the models using the testing data for the current fold and record the accuracy.
    3. After all folds have been used for testing, the script calculates the mean and standard deviation of the recorded accuracies for both models.
3. After all k values have been used for cross validation, the script plots the mean accuracy of each model against the k values.

The LeNet models are defined in separate files, `lenet5.py` and `lenetUpgraded.py`. These scripts define a function to create the respective models and (in case of the upgraded LeNet) a function to create the optimizer. 

## Result Interpretation

The final output of the script will be a plot of mean accuracy against the number of folds (k). This plot will help you visualize how the choice of k affects the performance of the models. The standard deviation of the accuracies provides a measure of how much variation there is in the accuracy of each model.

Please note that the accuracies might differ between runs due to the stochastic nature of the algorithms and procedures used in the script.
