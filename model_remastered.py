import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch, torchvision
from torch import nn
from torch.utils.data import dataset, DataLoader, WeightedRandomSampler
from torchvision import models, datasets, transforms
from torchvision.models import ResNet34_Weights
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import MulticlassF1Score
from timeit import default_timer as timer
from pathlib import Path
from tqdm.auto import tqdm
from mlxtend.plotting import plot_confusion_matrix


# Configs
EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-4
SEED = 42
NUM_WORKERS = 0
INPUT_H = 560
INPUT_W = 700
COLOR_CHANNELS = 1
WEIGHT_DECAY = 5e-4
DATA_DIR = Path("data/package_4_class")
SAVE_PATH = Path("model")
FIG_PATH = Path("fig")


# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Build dataset
def build_dataset(data_dir: Path, 
                  train_transform: torchvision.transforms = None, 
                  val_transform: torchvision.transforms = None):

    """Build the ImageFolder dataset using the directory and transforms passed."""
    
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    train_set = datasets.ImageFolder(
        root=train_dir, 
        transform=train_transform, 
        target_transform=None
    )
    val_set = datasets.ImageFolder(
        root=val_dir, 
        transform=val_transform, 
        target_transform=None
    )

    return train_set, val_set


# Build DataLoader
def build_dataloader(train_set: datasets.ImageFolder, 
                     val_set: datasets.ImageFolder, 
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
    val_loader = DataLoader(
        dataset=val_set, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )

    return train_loader, val_loader


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
                padding=original_conv.padding, 
                bias=False # this conv layer is followed by BN (the biases counteract), so we don't need bias here; however when followed by ReLU we can keep bias
            )

            # Copy the weights from original conv layer to the new conv layer
            with torch.inference_mode():
                replace_conv.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))

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
    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        # 1. Forward pass
        y_logits = model(X)
        # y_pred_probs = torch.softmax(y_logits, dim=1)
        y_pred_label = torch.argmax(y_logits, dim=1)

        # 2. Calculate the one batch loss
        loss = loss_fn(y_logits, y)
        train_loss += loss.item() # Use item here to avoid mps tensor get into numpy and avoid extra comp   utation cost
        total_correct += (y_pred_label == y).sum().item()
        total_samples += len(y) # also y.size(0) torch.Size([4]) -> 4

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. loss backward (backpropagation)
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Adjust the accumulated loss and acc to per epoch
    train_loss /= len(train_loader)
    train_acc = total_correct / total_samples

    return train_loss, train_acc


# Validate for one epoch
def val_step(model: nn.Module, 
              val_loader: DataLoader, 
              loss_fn: nn.Module = nn.CrossEntropyLoss(), 
              num_classes: int = 1, 
              device = device):

    """Perform val step for one epoch."""

    # Set up val loss and val accuracy
    val_loss, val_acc = 0, 0

    # Set up val macro-f1
    f1_metric = MulticlassF1Score(
        num_classes=num_classes, 
        average="macro"
    ).to(device)
    f1_metric.reset()

    # Setup val pred labels for plotting (confusion matrix)
    val_pred_labels = []

    model.to(device)
    model.eval()
    with torch.inference_mode():
        for batch, (X, y) in enumerate(val_loader):
            # Put the data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward Pass
            val_logits = model(X)
            # val_pred_probs = torch.softmax(val_logits, dim=1)
            val_pred_label = torch.argmax(val_logits, dim=1)
            val_pred_labels.append(val_pred_label)
            # Update f1 metric by accumulating the results in each batch, and calculate them eventually after val step by compute()
            f1_metric.update(val_pred_label, y)

            # 2. Calculate the loss and acc
            loss = loss_fn(val_logits, y)
            val_loss += loss.item() # Use item here to avoid getting a mps tensor in numpy and prevent extra computation
            val_acc += (val_pred_label == y).sum().item() / len(y)

        # Adjust the accumulated val loss and acc to per epoch
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        # Compute the macro-f1
        val_macro_f1 = f1_metric.compute().item()

    # Concat the val pred labels from each batch in one epoch
    val_pred_labels = torch.cat(val_pred_labels)

    return val_loss, val_acc, val_pred_labels, val_macro_f1


# Trainer
def train(model: nn.Module, 
          train_loader: DataLoader, 
          val_loader: DataLoader, 
          loss_fn: nn.Module = nn.CrossEntropyLoss(), 
          optimizer = torch.optim.Optimizer,
          epochs: int = 2, 
          num_classes: int = 1, 
          device = device) -> dict[str, list]:

    """Perform training and valing loops and return results dictionary."""

    model.to(device)
    print(f"Training on: {next(model.parameters()).device}")

    # Save results dictionary for dataframe and plot
    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}

    # Setup macro f1 metric
    best_f1 = 0.0

    # for epoch in tqdm(range(epochs)):
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_step(
            model=model, 
            train_loader=train_loader, 
            loss_fn=loss_fn, 
            optimizer=optimizer, 
            device=device
        )
        # Validate
        val_loss, val_acc, val_pred_labels, val_f1 = val_step(
            model=model, 
            val_loader=val_loader, 
            loss_fn=loss_fn, 
            num_classes=num_classes, 
            device=device
        )
        # Save the results per epoch to results dict
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        results["val_f1"].append(val_f1)

        # Print out what's happenin'
        print(f"Epoch: {epoch + 1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f} | Val F1: {val_f1:.4f}")

        # Save the model if Macro-F1 improved
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_model(
                model=model, 
                save_path=SAVE_PATH, 
                model_name="best_f1_model.pth"
            )
            # Print out best result and save info
            print(f"Macro-F1 new best: {best_f1:.4f}. Model state dict saved.")


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
    plt.plot(epochs, results["val_loss"], label="Val Loss")
    plt.title("Loss")
    plt.xlabel("epochs")
    plt.legend()

    # Plot the accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, results["train_acc"], label="Train Acc")
    plt.plot(epochs, results["val_acc"], label="Val Acc")
    plt.title("Accuracy")
    plt.xlabel("epochs")
    plt.legend();


# Plot val confusion matrix
def plot_the_confusion_matrix(class_names: list[str], 
                          pred_label: torch.Tensor, 
                          targets: list, 
                          fig_name: str):

    """Plot confusion matrix on a given set."""

    confmat = ConfusionMatrix(task='multiclass', 
                              num_classes=len(class_names))
    confmat_tensor = confmat(preds=pred_label.cpu(), # these are on gpu (if available), so we put them on cpu
                             target=torch.Tensor(targets)) # it receives tensors (Tensors are created on cpu by default)
    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(), 
        class_names=class_names, 
        figsize=(8, 8)
    )

    # Save the conf mat
    plt.savefig(FIG_PATH / fig_name)


# Save and load model
def save_model(model: nn.Module, 
                     save_path: Path = SAVE_PATH, # you must put the default argument after the non-default argument
                     model_name: str = "best_model.pth"): 

    """Save the model to a given path in a given name."""

    # Make directory of the save path
    save_path.mkdir(parents=True, 
                    exist_ok=True)
    model_save_path = save_path / model_name

    # Save the model state dict
    torch.save(
        obj=model.state_dict(), 
        f=model_save_path
    )


# Load a saved model state dict
def load_model(model: nn.Module, 
               load_path: Path = SAVE_PATH):

    """Load a saved model state dict from a given path (specific model name in path required)."""

    return model.load_state_dict(torch.load(f=load_path, map_location="cpu")) # its not a model


# Build transforms
def build_transforms(input_height = INPUT_H, input_width = INPUT_W):
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # transform images to gray scale images
        transforms.Resize(size=(INPUT_H, INPUT_W)), 
        transforms.RandomRotation(degrees=4), 
        transforms.ColorJitter(brightness=0.08, contrast=0.08), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.5], std=[0.25]) # for gray scale data, let the data 0-centered (only if matches expectation), easier for neural networks to train
    ])
    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), 
        transforms.Resize(size=(INPUT_H, INPUT_W)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.5], std=[0.25])
    ])
    return train_transform, val_transform


# Main function
def main():
    # Set seed for reproducability
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.mps.manual_seed(SEED)
    np.random.seed(SEED)

    # Setup transforms
    train_transform, val_transform = build_transforms(INPUT_H, INPUT_W)

    # Setup training set and val set
    train_set, val_set = build_dataset(
        data_dir=DATA_DIR, 
        train_transform=train_transform, 
        val_transform=val_transform
    )

    ### Setup sampler using WeightedRandomSampler for better sample distribution for training
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
    train_loader, val_loader = build_dataloader(
        train_set=train_set, 
        val_set=val_set, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS, 
        sampler=sampler
    )

    # Print the dataset attributes
    print(f"Train samples: {len(train_set)}")
    print(f"Val samples: {len(val_set)}")

    # Instantiate the model
    model = ResNet34Remix(
        input_shape=COLOR_CHANNELS, # color_channels
        output_shape=len(train_set.classes)
    ).to(device)

    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        params=model.parameters(), 
        lr=LR, # for Adam, default is 1e-3
        weight_decay=WEIGHT_DECAY
    )

    # Time the training time
    print("----------------")
    start_time = timer()
    # Train
    results = train(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        loss_fn=loss_fn, 
        optimizer=optimizer, 
        epochs=EPOCHS, 
        num_classes=len(val_set.classes), 
        device=device
    )
    end_time = timer()
    print("----------------")
    print(f"Time used: {end_time - start_time :.3f} seconds.")

    # Plot the loss and accuracy results
    plot_loss_and_acc(results)

    ### Run a val step using the best val f1 model parameters
    # Instantiate the unload model and put it to target device
    loaded_model = ResNet34Remix(
        input_shape=COLOR_CHANNELS, 
        output_shape=len(val_set.classes)
    ).to(device)

    # Load model state dict
    load_model(
        model=loaded_model, 
        load_path=SAVE_PATH / "best_f1_model.pth"
    )

    # Run a val step (one epoch)
    val_result = val_step(
        model=loaded_model, 
        val_loader=val_loader, 
        loss_fn=loss_fn, 
        num_classes=len(val_set.classes), 
        device=device
    )
    val_pred_labels = val_result[2]

    # Plot the confusion matrix on val
    plot_the_confusion_matrix(
        class_names=val_set.classes, 
        pred_label=val_pred_labels, 
        targets=val_set.targets, 
        fig_name="Val Confustion Matrix.png"
    )
    
    # Block the window from closing after training finished
    plt.show(block=True)


if __name__ == "__main__":
    main()