import torch
import random
import matplotlib.pyplot as plt
import numpy as np
from torchinfo import summary
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassF1Score
from pathlib import Path
from model_remastered import ResNet34Remix
from tqdm.auto import tqdm
from model_remastered import plot_the_confusion_matrix
from sklearn.metrics import classification_report


# Configs
NUM_SAMPLES = 16
INPUT_H = 560
INPUT_W = 700
COLOR_CHANNELS = 1
RANDOM = True # Not using seed if RANDOM sets to True
SEED = 42

# Directory configs
DATA_DIR = Path("data/package_4_class")
TEST_DIR = DATA_DIR / "test"
MODEL_PATH = Path("model/best_f1_model_cuda.pth")
# MODEL_PATH = Path("checkpoints/checkpoints_resnet34_v6_round_final/best_f1_model.pth")
FIG_PATH = Path("fig/")


# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Make predictions
def make_predictions(model: nn.Module, 
                     data: list, 
                     device = device):

    """Make predictions with given data samples list."""

    pred_probs = []

    model.to(device)
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # Add a batch dimension to the sample for PyTorch (note: didn't manipulate the original sample here), and put it to target device
            sample = torch.unsqueeze(sample, dim=0).to(device)

            # Forward pass and get pred probs
            pred_logits = model(sample)
            pred_prob = torch.softmax(pred_logits, dim=1)
            pred_probs.append(pred_prob)


    return pred_probs


# Create random samples from dataset
def create_random_samples(pred_set: datasets, 
                          num_samples: int = 9):

    """Create random samples from the given dataset."""

    # Randomly select data from the predict dataset
    pred_samples = []
    pred_labels = []
    for sample, label in random.sample(list(pred_set), k=num_samples):
        pred_samples.append(sample)
        pred_labels.append(label)

    return pred_samples, pred_labels


# Plot predictions
def plot_predictions(pred_samples: list, 
                     pred_labels: list, 
                     test_pred_probs: list, 
                     class_names: list, 
                     nrows: int = 3, 
                     ncols: int = 3):

    """Plot the predictions using the given prediction probabilities and by iterating pred samples list."""

    # Setup figure size
    plt.figure(figsize=(nrows * 2.5, ncols * 2.5))

    for i, sample in tqdm(enumerate(pred_samples), desc="Plot Predictions"):
        # Create subplot
        plt.subplot(nrows, ncols, i + 1)

        # Show sample image
        img = sample.detach().cpu() * 0.25 + 0.5 # unnormalize and restore brightness
        plt.imshow(img.permute(1, 2, 0), cmap="gray") # remove the first batch dimension (batch_size=1 here)

        # Get pred class and ground truth class
        pred_label = torch.argmax(test_pred_probs[i], dim=1)
        pred_class = class_names[pred_label]
        truth_class = class_names[pred_labels[i]]

        # Get prediction confidence
        conf = test_pred_probs

        # Setup title
        title = f"Pred: {pred_class}"
        if pred_class == truth_class:
            color = "g"
        else:
            color = "r"
            title += f"\nTruth: {truth_class}"
        title += f"\nconf: {torch.max(test_pred_probs[i]) * 100 :.2f} %"
        plt.title(title, fontsize=9, c=color)
        plt.axis(False)
        
        # Save the figure
        plt.savefig(FIG_PATH / "Predictions.png", dpi=200)


# Load a saved model state dict
def load_model(model: nn.Module, 
               load_path: Path = MODEL_PATH):

    """Load a saved model state dict from a given path (specific model name in path required)."""

    checkpoint = torch.load(f=load_path, map_location="cpu")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        result = model.model.load_state_dict(state_dict) # its not a model
    else:
        state_dict = checkpoint
        result = model.load_state_dict(state_dict)

    return result


# Perform a test step (for one epoch)
def test_step(model: nn.Module, 
              test_loader: torch.utils.data.DataLoader, 
              num_classes: int, 
              class_names: list[str], 
              device = device) -> tuple:

    """Run a test step using model and test set."""

    # Setup test metrics
    test_correct, test_total = 0, 0
    f1_metric = MulticlassF1Score(
        num_classes=num_classes, 
        average="macro"
    ).to(device)
    f1_metric.reset()
    test_pred_labels = []
    test_true_labels = []

    model.to(device)
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(test_loader, desc="Test"):
            # Put the data to the target device
            X, y = X.to(device), y.to(device)

            # Forward pass
            test_logits = model(X)
            test_pred_class = torch.argmax(test_logits, dim=1)
            test_pred_labels.append(test_pred_class)
            test_true_labels.append(y)

            # Update the correct sample count for acc computation
            test_correct += (test_pred_class == y).sum().item()
            test_total += len(y)
            # Update the f1 score
            f1_metric.update(test_pred_class, y)

        # Prepare the results
        test_acc = test_correct / test_total
        test_f1 = f1_metric.compute().item()
        test_pred_labels = torch.cat(test_pred_labels, dim=0).cpu() # use torch.cat() to get a 1D tensor, concat along the sample dimension
        test_true_labels = torch.cat(test_true_labels, dim=0).cpu()
        report = classification_report(
            y_pred=test_pred_labels, 
            y_true=test_true_labels, 
            labels=range(num_classes), # range turns a number (say 4) into numbers (like 0, 1, 2, 3)
            target_names=class_names, 
            zero_division=0
        )

    # Return the results
    return test_acc, test_f1, test_pred_labels, report


# Set random seed
def set_random_seed(seed = SEED):
    torch.manual_seed(seed)
    random.seed(seed)


# Main function
def main():
    # Set random seed if random is set to false
    if not RANDOM:
        set_random_seed(SEED)

    # Setup test data transform
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), 
        transforms.Resize(size=(INPUT_H, INPUT_W)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.5], std=[0.25]) # normalize for gray scale images
    ])

    # Setup test dataset
    test_set = datasets.ImageFolder(
        root=TEST_DIR, 
        transform=test_transform, 
        target_transform=None
    )

    # Setup test loader
    test_loader = DataLoader(
        dataset=test_set, 
        batch_size=1, 
        num_workers=0, 
        # pin_memory=True, # enable on cuda devices
        # persistent_workers=True
    )

    # Create random prediction samples and labels
    pred_samples, pred_labels = create_random_samples(
        pred_set=test_set, 
        num_samples=NUM_SAMPLES
    )

    # Instantiate new model
    loaded_model = ResNet34Remix(
        input_shape=COLOR_CHANNELS, # color channels
        output_shape=len(test_set.classes)
    ).to(device)

    # Load model state dict
    load_model(
        model=loaded_model, 
        load_path=MODEL_PATH
    )

    # Make predictions with the loaded model
    pred_probs = make_predictions(
        model=loaded_model, 
        data=pred_samples, 
        device=device
    )

    # Plot the predictions
    ncols = int(np.ceil(np.sqrt(NUM_SAMPLES)))
    nrows = int(np.ceil(NUM_SAMPLES / ncols))
    plot_predictions(
        pred_samples=pred_samples, 
        pred_labels=pred_labels, 
        test_pred_probs=pred_probs, 
        class_names=test_set.classes, 
        nrows=nrows, 
        ncols=ncols
    )

    # Run a full test and evaluate
    test_acc, test_f1, test_pred_labels, report = test_step(
        model=loaded_model, 
        test_loader=test_loader, 
        num_classes=len(test_set.classes), 
        class_names=test_set.classes, 
        device=device
    )

    # Print out the results
    print(f"Test Acc: {test_acc:.4f} | Test Macro-F1: {test_f1:.4f}")
    print(report)

    # Plot confusion matrix
    plot_the_confusion_matrix(
        class_names=test_set.classes, 
        pred_label=test_pred_labels, # Tensors
        targets=test_set.targets, 
        fig_name="Test Confusion Matrix.png"
    )

    # Print the summary of the model
    summary(
        model=loaded_model, 
        input_size=[1, COLOR_CHANNELS, INPUT_H, INPUT_W]
    )

    # Display all open figures and prevent closing
    # plt.show(block=True)


if __name__ == "__main__":
    main()