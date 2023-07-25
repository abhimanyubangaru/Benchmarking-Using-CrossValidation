import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch import optim 
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from lenet5 import create_lenet
from lenetUpgraded import create_upgraded_lenet, get_optimizer_for_upgraded_lenet

# Load MNIST Data
train_dataset = datasets.MNIST('./mnist_data', download=True, train=True, transform=transforms.ToTensor())

x_train = train_dataset.data
y_train = train_dataset.targets

x_train = x_train.unsqueeze(1).float() / 255.0  # Add channel dimension and normalize

# List of k-values
k_values = [2,5,10]

accuracy_values_lenet5 = []
accuracy_values_upgraded_lenet = []

for k in k_values:
    print(f'K-Fold Cross Validation with k={k}')
    
    # Define k-fold cross validation
    kfold = KFold(n_splits=k, shuffle=True)

    cvscores_lenet5 = []
    cvscores_upgraded_lenet = []

    for fold, (train_ids, test_ids) in enumerate(kfold.split(x_train)):
        print(f'FOLD {fold}')
        print('--------------------------------')

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, sampler=test_subsampler)

        modelLenet5 = create_lenet()
        modelUpgradedLenet = create_upgraded_lenet()

        criterion = nn.CrossEntropyLoss()

        optimizer_lenet5 = optim.Adam(modelLenet5.parameters(),lr=1e-3)
        optimizer_upgraded_lenet = get_optimizer_for_upgraded_lenet(modelUpgradedLenet)

        models = [(modelLenet5, optimizer_lenet5, cvscores_lenet5,accuracy_values_lenet5), (modelUpgradedLenet, optimizer_upgraded_lenet, cvscores_upgraded_lenet,accuracy_values_upgraded_lenet)]

        for model, optimizer, cvscores, accuracy_scores in models:
            # Training loop
            for epoch in range(10): 
                running_loss = 0.0
                for i, (inputs, labels) in enumerate(trainloader, 0):
                    inputs, labels = Variable(inputs), Variable(labels)

                    optimizer.zero_grad()

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

            # Evaluation for this fold
            correct, total = 0, 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100.0 * correct / total
            print('Accuracy for fold %d: %.2f %%' % (fold, accuracy))
            cvscores.append(accuracy)
            print('--------------------------------')

    mean_accuracy_for_l5 = np.mean(cvscores_lenet5)
    mean_accuracy_for_lU = np.mean(cvscores_upgraded_lenet)
    accuracy_values_lenet5.append(mean_accuracy_for_l5)
    accuracy_values_upgraded_lenet.append(mean_accuracy_for_lU)
    print('Final K-Fold Cross Validation Accuracy for Lenet 5 k=%d: %.2f%% (+/- %.2f%%)' % (k, mean_accuracy_for_l5, np.std(cvscores_lenet5)))
    print('Final K-Fold Cross Validation Accuracy for Lenet Upgraded k=%d: %.2f%% (+/- %.2f%%)' % (k, mean_accuracy_for_lU, np.std(cvscores_upgraded_lenet)))
    print('======================================================')

# Plotting
plt.figure(figsize=(12, 6))
print(accuracy_values_lenet5)
print(accuracy_values_upgraded_lenet)
plt.plot(k_values, accuracy_values_lenet5, marker='o', label='LeNet5')
plt.plot(k_values, accuracy_values_upgraded_lenet, marker='o', label='Upgraded LeNet')
plt.title('Mean Accuracy vs. k')
plt.xlabel('Number of Folds (k)')
plt.ylabel('Mean Accuracy (%)')
plt.legend()
plt.grid(True)
plt.show()
