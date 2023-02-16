"""Utilities for nah calculation"""
# %%
from typing import Optional, Sequence

import einops
import numpy as np
from sklearn.metrics import mutual_info_score  # type: ignore
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm as tqdm

# %%
# converts a possibly torch tensor or list (on either cpu or "cuda") to a numpy array
def ensure_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)


def image_to_int_np(
    image_batch: np.ndarray, nbins: int = 5, use_object: bool = False
) -> np.ndarray:
    """Convert a batch of images to ints."""
    if use_object:
        image_batch = np.array(image_batch, dtype=object)
    image_batch = einops.rearrange(image_batch, "b h w -> b (h w)")
    if not use_object and image_batch.shape[1] > 64 / np.log2(nbins):
        print("Warning: image is too large to be converted to an int, will overflow. ")
    return np.sum(image_batch * nbins ** np.arange(image_batch.shape[1]), axis=1)


def image_to_int_torch(image_batch, nbins=5) -> torch.Tensor:
    """Convert a batch of images in a torch tensor to ints."""
    image_batch = einops.rearrange(image_batch, "b h w -> b (h w)")
    if image_batch.shape[1] > 64 / np.log2(nbins):
        print("Warning: image is too large to be converted to an int, will overflow. ")
    return torch.sum(image_batch * nbins ** torch.arange(image_batch.shape[1]), dim=1)


def features_to_int_np(
    features_batch: np.ndarray, nbins: int = 5, use_object: bool = False
) -> np.ndarray:
    """Convert a batch of features to ints."""
    if use_object:
        features_batch = np.array(features_batch, dtype=object)
    if not use_object and features_batch.shape[1] > 64 / np.log2(nbins):
        print("Warning: image is too large to be converted to an int, will overflow. ")
    return np.sum(features_batch * nbins ** np.arange(features_batch.shape[1]), axis=1)


def mi_one_pixel_removed(
    image_batch: Sequence[np.ndarray], target: Sequence[int]
) -> np.ndarray:
    """
    Compute the mutual information of the remainder of the image with target after removing each pixel.

    Args:
        image_batch: a batch of int images of shape (batch_size, height, width)
        target: a batch of int targets of shape (batch_size)

    Returns:
        mi: a matrix of shape (height, width) containing the mutual information after removing each pixel.
    """
    _, height, width = image_batch.shape
    mi = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            image_batch_removed = image_batch.copy()
            image_batch_removed[:, i, j] = 0
            image_batch_removed_int = image_to_int_np(image_batch_removed)
            mi[i, j] = mutual_info_score(image_batch_removed_int, target)

    return mi


def mi_one_feature_removed(
    image_batch: Sequence[np.ndarray], target: Sequence[int]
) -> np.ndarray:
    """
    Compute the mutual information of the remainder of the image with target after removing each feature.

    Args:
        image_batch: a batch of int images of shape (batch_size, n_features)
        target: a batch of int targets of shape (batch_size)

    Returns:
        mi: a matrix of shape (n_features) containing the mutual information after removing each feature.
    """
    _, n_features = image_batch.shape
    mi = np.zeros(n_features)

    for i in range(n_features):
        image_batch_removed = image_batch.copy()
        image_batch_removed[:, i] = 0
        image_batch_removed_int = features_to_int_np(image_batch_removed)
        mi[i] = mutual_info_score(image_batch_removed_int, target)

    return mi


# %%
class SmallConvNet(nn.Module):
    def __init__(self, feature_dim: int = 64):
        super(SmallConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(7 * 7 * 64, feature_dim)
        self.fc2 = nn.Linear(feature_dim, 10)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def predict(self, x: torch.Tensor):
        return torch.argmax(self.forward(x), dim=1)

    def get_features(self, x: torch.Tensor):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    n_epochs: int = 5,
    device: Optional[torch.device] = None,
    report_test_accuracy_every=1,
) -> None:
    """Trains a model to minimize cross entropy loss on the training set and reports the test accuracy every epoch."""
    for epoch in range(1, n_epochs + 1):
        for batch_idx, (data, target) in enumerate(tqdm.tqdm(train_loader)):
            if device is not None:
                data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                )
        if epoch % report_test_accuracy_every == 0:
            loss_test, acc_test = test(model, device, test_loader)
            print(f"Test set: Average loss: {loss_test:.4f}, Accuracy: {acc_test:.0f}%")


def test(
    model: nn.Module, test_loader: DataLoader, device: Optional[torch.device] = None
):
    """Computes the cross entropy loss and accuracy on the test set."""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if device is not None:
                data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100.0 * correct / len(test_loader.dataset)
    return test_loss, acc


# %%
class lookuptable:
    """Class that implements a lookup table for a given dataset. Like k-nearest neighbors, but for only exact matches."""

    def __init__(
        self,
    ):
        self.counts_table = dict()
        self.lookup_table = dict()

    def fit(self, X, y):
        for x, y in zip(X, y):
            x = tuple(x)
            if x not in self.counts_table:
                self.counts_table[x] = 0
                self.lookup_table[x] = y
            self.counts_table[x] += 1

    def __call__(self, x):
        return self.table[x]
