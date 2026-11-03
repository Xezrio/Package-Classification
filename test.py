import torch
import random
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torchvision import datasets, transforms
from pathlib import Path
from model_remastered import ResNet34Remix


# Configs
NUM_SAMPLES = 16
INPUT_H = 560
INPUT_W = 700
COLOR_CHANNELS = 1
RANDOM = True
SEED = 42
DATA_DIR = Path("data/package_4_class")
PRED_DIR = DATA_DIR / "val"
MODEL_PATH = Path("checkpoints/checkpoints_resnet34_v6_round_final/best_f1_model.pth")


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

    for i, sample in enumerate(pred_samples):
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


# Load a saved model state dict
def load_model(model: nn.Module, 
               load_path: Path = MODEL_PATH):

    """Load a saved model state dict from a given path (specific model name in path required)."""

    checkpoint = torch.load(f=load_path, map_location="cpu")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    return model.model.load_state_dict(state_dict) # its not a model


# Set random seed
def set_random_seed(seed = SEED):
    torch.manual_seed(seed)
    random.seed(seed)


# Main function
def main():
    # Set random seed if random is set to false
    if not RANDOM:
        set_random_seed(SEED)

    # Setup pred data transform
    pred_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), 
        transforms.Resize(size=(INPUT_H, INPUT_W)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.5], std=[0.25]) # normalize for gray scale images
    ])

    # Setup pred dataset
    pred_set = datasets.ImageFolder(
        root=PRED_DIR, 
        transform=pred_transform
    )

    # Create random prediction samples and labels
    pred_samples, pred_labels = create_random_samples(
        pred_set=pred_set, 
        num_samples=NUM_SAMPLES
    )

    # Instantiate new model
    loaded_model = ResNet34Remix(
        input_shape=COLOR_CHANNELS, # color channels
        output_shape=len(pred_set.classes)
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
        class_names=pred_set.classes, 
        nrows=nrows, 
        ncols=ncols
    )

    # Display all open figures and prevent closing
    plt.show(block=True)


if __name__ == "__main__":
    main()