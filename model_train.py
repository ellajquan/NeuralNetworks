import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import numpy as np
from ptflops import get_model_complexity_info
from torchsummary import summary
from models import vgg, resnet  # Adjust the path if needed
from utils import progress_bar  # Assuming `progress_bar` is defined in utils.py
import time

# Set up the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Function to prepare data loaders
def prepare_data():
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    return trainloader, testloader

# Function to define the models
def create_models():
    net_vgg = vgg.VGG('VGG11').to(device)
    net_resnet = resnet.ResNet18().to(device)
    return net_vgg, net_resnet

# Calculate Parameters and MACs for a given model
def compute_model_statistics(model, model_name):
    # Number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"{model_name} - Number of Parameters: {num_params}")

    # Use ptflops to calculate MACs
    macs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True, print_per_layer_stat=False)
    print(f'{model_name} - MACs: {macs}, Parameters: {params}')

    return num_params, macs

# Test Accuracy Calculation
def test_accuracy(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_accuracy = 100. * correct / total
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    return test_accuracy

# Training function with time tracking and progress bar
def train(epoch, net, optimizer, trainloader, criterion, train_losses, train_accuracies):
    print(f'\nEpoch: {epoch}')
    net.train()
    train_loss, correct, total = 0, 0, 0
    epoch_start_time = time.time()  # Record start time of the epoch
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        batch_start_time = time.time()  # Start time for each batch

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Progress bar update
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        # Print batch processing time
        print(f"Batch {batch_idx + 1}/{len(trainloader)} took {time.time() - batch_start_time:.2f} seconds")

    # Append average training loss and accuracy for this epoch
    train_losses.append(train_loss / (batch_idx + 1))
    train_accuracies.append(100. * correct / total)

    # Print epoch processing time
    print(f"Epoch {epoch} training took {time.time() - epoch_start_time:.2f} seconds")


# Testing function with time tracking and progress bar
def test(epoch, net, testloader, criterion, test_losses, test_accuracies):
    net.eval()
    test_loss, correct, total = 0, 0, 0
    epoch_start_time = time.time()  # Record start time of the epoch
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Progress bar update
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Append average test loss and accuracy for this epoch
    test_losses.append(test_loss / (batch_idx + 1))
    test_accuracies.append(100. * correct / total)

    # Print epoch processing time
    print(f"Epoch {epoch} testing took {time.time() - epoch_start_time:.2f} seconds")


import os

# Save sample images with predictions
def save_sample_images(model, dataloader, save_path='sample_predictions.png', num_images=5):
    # Create the directory if it does not exist
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Get a batch of images and labels
    dataiter = iter(dataloader)
    images, labels = next(dataiter)

    # Get predictions from the model
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    # Show the first `num_images` images and predictions
    fig, axes = plt.subplots(1, num_images, figsize=(12, 6))
    for i in range(num_images):
        ax = axes[i]
        image = images[i].cpu().numpy().transpose((1, 2, 0))  # Convert to (H, W, C) format for display
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        image = std * image + mean  # Unnormalize the image
        image = np.clip(image, 0, 1)  # Clip the values to ensure valid pixel range
        ax.imshow(image)
        ax.set_title(f"Predicted: {predicted[i].item()}")
        ax.axis('off')

    plt.suptitle(f"Sample Images - Predicted: {predicted[:num_images].cpu().numpy()}")
    
    # Save the plot
    plt.savefig(save_path)
    print(f"Sample images saved to: {save_path}")
    plt.close()

# Function to plot and save metrics
def plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies, model_name, save_dir='plots'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{model_name} - Loss')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title(f'{model_name} - Accuracy')

    plot_filename = os.path.join(save_dir, f"{model_name}_metrics.png")
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")
    plt.close()

# Save model statistics to a text file
def save_statistics_to_file(stats, filename='model_statistics.txt'):
    with open(filename, 'w') as f:
        for model_name, (params, macs, accuracy) in stats.items():
            f.write(f"{model_name} - Parameters: {params}, MACs: {macs}, Test Accuracy: {accuracy:.2f}%\n")

# Main function for running the entire script
def main():
    trainloader, testloader = prepare_data()

    # Initialize models
    net_vgg, net_resnet = create_models()

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_vgg = optim.SGD(net_vgg.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    optimizer_resnet = optim.SGD(net_resnet.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    # Define learning rate scheduler
    scheduler_vgg = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_vgg, T_max=150)
    scheduler_resnet = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_resnet, T_max=150)

    # Initialize metric storage
    train_losses_vgg, train_accuracies_vgg = [], []
    test_losses_vgg, test_accuracies_vgg = [], []

    train_losses_res, train_accuracies_res = [], []
    test_losses_res, test_accuracies_res = [], []

    # Train VGG11
    print("==> Training VGG11 model..")
    for epoch in range(151):
        train(epoch, net_vgg, optimizer_vgg, trainloader, criterion, train_losses_vgg, train_accuracies_vgg)
        test(epoch, net_vgg, testloader, criterion, test_losses_vgg, test_accuracies_vgg)
        scheduler_vgg.step()

    torch.save(net_vgg.state_dict(), 'vgg11_cifar10.pth')

    # Train ResNet18
    print("==> Training ResNet18 model..")
    for epoch in range(151):
        train(epoch, net_resnet, optimizer_resnet, trainloader, criterion, train_losses_res, train_accuracies_res)
        test(epoch, net_resnet, testloader, criterion, test_losses_res, test_accuracies_res)
        scheduler_resnet.step()
    

    torch.save(net_resnet.state_dict(), 'resnet18_cifar10.pth')

    # Compute and save model statistics
    model_statistics = {}
    print("==> Calculating metrics for VGG11..")
    vgg_params, vgg_macs = compute_model_statistics(net_vgg, "VGG11")
    vgg_test_accuracy = test_accuracy(net_vgg, testloader)
    model_statistics["VGG11"] = (vgg_params, vgg_macs, vgg_test_accuracy)

    print("==> Calculating metrics for ResNet18..")
    resnet_params, resnet_macs = compute_model_statistics(net_resnet, "ResNet18")
    resnet_test_accuracy = test_accuracy(net_resnet, testloader)
    model_statistics["ResNet18"] = (resnet_params, resnet_macs, resnet_test_accuracy)

    save_statistics_to_file(model_statistics)

    # Plot metrics
    plot_metrics(train_losses_vgg, test_losses_vgg, train_accuracies_vgg, test_accuracies_vgg, "VGG11")
    plot_metrics(train_losses_res, test_losses_res, train_accuracies_res, test_accuracies_res, "ResNet18")

    # Save sample images with predictions
    save_sample_images(net_vgg, testloader, save_path='/home/ellaquan/NeuralNetworks/results/sample_vgg11_predictions.png')
    save_sample_images(net_resnet, testloader, save_path='/home/ellaquan/NeuralNetworks/results/sample_resnet18_predictions.png')

if __name__ == "__main__":
    main()