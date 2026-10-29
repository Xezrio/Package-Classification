import numpy as np
import pandas as pd
import torch, torchvision, torchmetrics
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import dataset, DataLoader, WeightedRandomSampler
from torchvision import models, datasets, transforms
from torchvision.models import ResNet34_Weights
from timeit import default_timer as timer
from pathlib import Path
from tqdm.auto import tqdm
from helper_functions import print_dataset_attributes
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix


# Configs
EPOCHS = 8
BATCH_SIZE = 32
LR = 0.001
SEED = 42
NUM_WORKERS = 0
INPUT_H = 224
INPUT_W = 224
DATA_DIR = Path("data/pizza_steak_sushi")


# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Build dataset
def build_dataset(data_dir: Path, 
                  train_transform: torchvision.transforms = None, 
                  test_transform: torchvision.transforms = None):

    """Build the ImageFolder dataset using the directory and transforms passed."""
    
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    train_set = datasets.ImageFolder(
        root=train_dir, 
        transform=train_transform, 
        target_transform=None
    )
    test_set = datasets.ImageFolder(
        root=test_dir, 
        transform=test_transform, 
        target_transform=None
    )

    return train_set, test_set


# Build DataLoader
def build_dataloader(train_set: datasets.ImageFolder, 
                     test_set: datasets.ImageFolder, 
                     batch_size: int, 
                     num_workers: int = 0, 
                     sampler = None):

    """Build the dataloaders using the datasets."""

    train_loader = DataLoader(
        dataset=train_set, 
        batch_size=batch_size, 
        # shuffle=True, # if sampler is used then shuffle is not available 
        num_workers=num_workers, 
        sampler=sampler
    )
    test_loader = DataLoader(
        dataset=test_set, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )

    return train_loader, test_loader


# ResNet34 Model
class ResNet34Remix(nn.Module):
    def __init__(self, 
                 input_shape: int, 
                 output_shape: int):
        
        super().__init__()

        self.model = models.resnet34(
            weights=ResNet34_Weights.DEFAULT
        )
        self.model.fc = nn.Linear(
            in_features=self.model.fc.in_features, 
            out_features=output_shape
        )

        if input_shape != self.model.conv1.in_channels:
            original_conv = self.model.conv1 # the original resnet34 first conv layer
            replace_conv = nn.Conv2d(
                in_channels=input_shape, 
                out_channels=original_conv.out_channels, 
                kernel_size=original_conv.kernel_size, 
                stride=original_conv.stride, 
                padding=original_conv.padding
            )

            # Copy the weights from original conv layer to the new conv layer
            with torch.inference_mode():
                replace_conv.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True)) # ?

            self.model.conv1 = replace_conv
            
    def forward(self, x):
        return self.model(x)


# Train for one epoch
def train_step(model: nn.Module, 
               train_loader: DataLoader, 
               loss_fn: nn.Module = nn.CrossEntropyLoss(), 
               optimizer = torch.optim.Optimizer, 
               device = device):

    """Perform train step for one epoch."""

    # Setup train loss and train acc
    train_loss, train_acc = 0, 0

    model.to(device)
    model.train()
    for batch, (X, y) in tqdm(enumerate(train_loader)):
        X, y = X.to(device), y.to(device)
        # 1. Forward pass
        y_logits = model(X)
        y_pred_probs = torch.softmax(y_logits, dim=1)
        y_pred_label = torch.argmax(y_pred_probs, dim=1)

        # 2. Calculate the one batch loss
        loss = loss_fn(y_logits, y)
        train_loss += loss.item() # Use item here to avoid mps tensor get into numpy and avoid extra computation cost
        train_acc += (y_pred_label == y).sum().item() / len(y)

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. loss backward (backpropagation)
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Adjust the accumulated loss and acc to per epoch
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    return train_loss, train_acc


# Test for one epoch
def test_step(model: nn.Module, 
              test_loader: DataLoader, 
              loss_fn: nn.Module = nn.CrossEntropyLoss(), 
              device = device):

    """Perform test step for one epoch."""

    # Set up test loss and test accuracy
    test_loss, test_acc = 0, 0

    # Setup test pred labels for plotting (confusion matrix)
    test_pred_labels = []

    model.to(device)
    model.eval()
    with torch.inference_mode():
        for batch, (X, y) in tqdm(enumerate(test_loader)):
            # Put the data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward Pass
            test_logits = model(X)
            test_pred_probs = torch.softmax(test_logits, dim=1)
            test_pred_label = torch.argmax(test_pred_probs, dim=1)
            test_pred_labels.append(test_pred_label)

            # 2. Calculate the loss and acc
            loss = loss_fn(test_logits, y)
            test_loss += loss.item() # Use item here to avoid getting a mps tensor in numpy and prevent extra computation
            test_acc += (test_pred_label == y).sum().item() / len(y)

        # Adjust the accumulated test loss and acc to per epoch
        test_loss /= len(test_loader)
        test_acc /= len(test_loader)

    # Concat the test pred labels from each batch in one epoch
    test_pred_labels = torch.cat(test_pred_labels)

    return test_loss, test_acc, test_pred_labels


# Trainer
def train(model: nn.Module, 
          train_loader: DataLoader, 
          test_loader: DataLoader, 
          loss_fn: nn.Module = nn.CrossEntropyLoss(), 
          optimizer = torch.optim.Optimizer, # ?
          epochs: int = 2, 
          device = device) -> dict[str, list]:

    """Perform training and testing loops and return results dictionary."""

    model.to(device)
    print(f"Training on: {next(model.parameters()).device}")

    # Save results dictionary for dataframe and plot
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in tqdm(range(epochs)):
        # Train
        train_loss, train_acc = train_step(
            model=model, 
            train_loader=train_loader, 
            loss_fn=loss_fn, 
            optimizer=optimizer, 
            device=device
        )
        # Test
        test_loss, test_acc, test_pred_labels = test_step(
            model=model, 
            test_loader=test_loader, 
            loss_fn=loss_fn, 
            device=device
        )
        # Save the results per epoch to results dict
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        # Print out what's happenin'
        print(f"Epoch: {epoch + 1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}")

    return results


# Plot the results
def plot_loss_and_acc(results: dict[str, list]):

    """Plot the loss and accuracy results."""

    # plt.show() is called in the main function with block=True

    # Get a range epoch num
    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(10, 5))

    # Plot the loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, results["train_loss"], label="Train Loss")
    plt.plot(epochs, results["test_loss"], label="Test Loss")
    plt.title("Loss")
    plt.xlabel("epochs")
    plt.legend()

    # Plot the accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, results["train_acc"], label="Train Acc")
    plt.plot(epochs, results["test_acc"], label="Test Acc")
    plt.title("Accuracy")
    plt.xlabel("epochs")
    plt.legend();



# Plot test confusion matrix
def plot_test_confusion_matrix(class_names: list[str], 
                          test_pred_label: torch.Tensor, 
                          test_targets: list):

    """Plot confusion matrix on test set."""

    confmat = ConfusionMatrix(task='multiclass', 
                              num_classes=len(class_names))
    confmat_tensor = confmat(preds=test_pred_label.cpu(), # these are on gpu (if available), so we put them on cpu
                             target=torch.Tensor(test_targets)) # it receives tensors (Tensors are created on cpu by default)
    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(), 
        class_names=class_names, 
        figsize=(7, 7)
    )


# Build transforms
def build_transforms(input_height = INPUT_H, input_width = INPUT_W):
    train_transform = transforms.Compose([
        transforms.Resize(size=(INPUT_H, INPUT_W)), 
        transforms.RandomRotation(degrees=4), 
        transforms.ColorJitter(brightness=0.1, contrast=0.1), 
        transforms.ToTensor(), 
        # transforms.Normalize(mean=[0.5], std=[0.25]) # for gray scale data, let the data 0-centered, easier for neural networks to train
    ])
    test_transform = transforms.Compose([
        transforms.Resize(size=(INPUT_H, INPUT_W)), 
        transforms.ToTensor()
    ])
    return train_transform, test_transform


# Main function
def main():
    # Set seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.mps.manual_seed(SEED)
    np.random.seed(SEED)

    # Setup transforms
    train_transform, test_transform = build_transforms(INPUT_H, INPUT_W)

    # Setup training set and test set
    train_set, test_set = build_dataset(
        data_dir=DATA_DIR, 
        train_transform=train_transform, 
        test_transform=test_transform
    )

    # Setup sampler using WeightedRandomSampler for better sample distribution for training
    targets = train_set.targets # it looks like [0, 0, 0, 1, 1, ..., 1, 2, 2]
    class_counts = torch.bincount(torch.tensor(targets)) # pass in a tensor, you get a tensor shaped like tensor([78, 75, 72])
    class_weights = (1. / class_counts).numpy() # turn tensor into numpy ndarray [1/78, 1/75, 1/72]
    # Assign weight to each sample
    sample_weights = class_weights[targets] # map: class_weights[use targets[] as indexes]

    sampler = WeightedRandomSampler(
        weights=sample_weights,  
        num_samples=len(train_set), 
        replacement=True # replacement=True means the sample would be put back after it's chosen
    )

    # Setup dataloaders
    train_loader, test_loader = build_dataloader(
        train_set=train_set, 
        test_set=test_set, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS, 
        sampler=sampler
    )

    # Check the dataset attributes
    print_dataset_attributes(
        train_set=train_set, 
        test_set=test_set
    )

    # Instantiate the model
    model = ResNet34Remix(
        input_shape=3, 
        output_shape=len(train_set.classes)
    )

    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), 
        lr=LR # for Adam, default is 1e-3
    )

    # Time the training time
    print("----------------")
    start_time = timer()
    # Train
    results = train(
        model=model, 
        train_loader=train_loader, 
        test_loader=test_loader, 
        loss_fn=loss_fn, 
        optimizer=optimizer, 
        epochs=EPOCHS, 
        device=device
    )
    end_time = timer()
    print("----------------")
    print(f"Time used: {end_time - start_time :.3f} seconds.")

    # Plot the loss and accuracy results
    plot_loss_and_acc(results)

    # Run a test step using the latest model parameters
    test_result = test_step(
        model=model, 
        test_loader=test_loader, 
        loss_fn=loss_fn, 
        device=device
    )
    test_pred_labels = test_result[2]

    # Plot the test confusion matrix
    plot_test_confusion_matrix(
        class_names=test_set.classes, 
        test_pred_label=test_pred_labels, 
        test_targets=test_set.targets
    )

    # Block the window from closing after training finished
    plt.show(block=True)


if __name__ == "__main__":
    main()