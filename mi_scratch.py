""" A VSCode notebook that tries to estimate the empirical mutual information curves."""
# %%
import einops
import numpy as np
import matplotlib.pyplot as plt
import plotly  # type: ignore
import plotly.express as px  # type: ignore
import scipy
from sklearn.metrics import mutual_info_score  # type: ignore
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# %%
# seed torch and numpy
torch.manual_seed(1)
np.random.seed(1)

# %%
# first, check if mutual_info_score is working
X = [0, 1, 0, 1]
Y = [0, 0, 1, 1]

mutual_info_score(X, Y)  # should be 0
# %%
X = [0, 1, 0, 1]
Y = [1, 2, 1, 2]

mutual_info_score(X, Y)  # should be ln(2) ~= 0.693
# %%
# import mnist data from torch
mnist_trainset = datasets.MNIST("./data", train=True, download=True, transform=None)
train_subset, val_subset = torch.utils.data.random_split(
    mnist_trainset, [50000, 10000], generator=torch.Generator()
)
# %%
X_val = val_subset.dataset.data[val_subset.indices]  # type: ignore
y_val = val_subset.dataset.targets[val_subset.indices]  # type: ignore

# %%
# display a few images
fig, ax = plt.subplots(1, 10, figsize=(10, 1))
for i in range(10):
    ax[i].imshow(X_val[i], cmap="gray")  # type: ignore
    ax[i].axis("off")  # type: ignore
plt.show()
# %%
# downsample X_val to 9 x 9 images using max pooling
X_val = X_val.reshape(-1, 1, 28, 28).to(torch.float32)
X_val = torch.nn.MaxPool2d(3)(X_val)
X_val = X_val.reshape(-1, 9, 9)

# %%
# now discretize the images into 5 values
X_val = (X_val / (255 / 4) + 0.5).to(torch.int)
# %%
# display a few images
fig, ax = plt.subplots(1, 10, figsize=(10, 1))
for i in range(10):
    ax[i].imshow(X_val[i], cmap="gray")  # type: ignore
    ax[i].axis("off")  # type: ignore
plt.show()
# %%
# now compute the mutual information between each pixel and the label
mi = np.zeros((9, 9))
for i in range(9):
    for j in range(9):
        mi[i, j] = mutual_info_score(X_val[:, i, j], y_val)
# %%
# plot the mutual information
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.imshow(mi, cmap="gray")
ax.set_title("Mutual Information")
ax.set_xlabel("Pixel Column")
ax.set_ylabel("Pixel Row")
plt.show()

# %%
# plot mutual information by rank of pixel
mi_rank = np.argsort(mi.flatten())[::-1]
mi_rank = mi.flatten()[mi_rank]
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(mi_rank)
ax.set_title("Mutual Information by Rank")
ax.set_xlabel("Pixel Rank")
ax.set_ylabel("Mutual Information")
plt.show()
# %%

# compute the total mutual information by converting each image into an int


# def image_to_int(image):
#     """Convert an image to an int."""
#     image = image.reshape(-1)
#     return int(torch.sum(image * 5 ** torch.arange(len(image))))


def image_to_int(image_batch):
    """Convert a batch of images to ints."""
    image_batch = einops.rearrange(image_batch, "b h w -> b (h w)")
    return torch.sum(image_batch * 5 ** torch.arange(image_batch.shape[1]), dim=1)


# %%
# compute the total mutual information by converting each image into an int
X_val_int = image_to_int(X_val)
mi_total = mutual_info_score(X_val_int, y_val)
print(f"Total mutual information: {mi_total:.3f} out of {np.log(10):.3f}")
# %%
# for each of the 81 pixels, delete that pixel and compute the mutual information of the remainder


def mi_one_pixel_removed(X_val, y_val):
    """
    Compute the mutual information of the remainder of the image after removing each pixel.
    """
    mi = np.zeros((9, 9))
    for i in range(9):
        for j in range(9):
            X_val_removed = X_val.clone()
            X_val_removed[:, i, j] = 0
            X_val_removed_int = image_to_int(X_val_removed)
            mi[i, j] = mutual_info_score(X_val_removed_int, y_val)

    return mi


def entropy(x, axis=0, eps=1e-15):
    """Compute the entropy of a discrete random variable along the given axis."""
    n_bins = np.max(x) + 1
    counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=n_bins), axis, x)
    probs = counts / np.sum(counts, axis=axis, keepdims=True) + eps
    return -np.sum(probs * np.log(probs), axis=axis)


# %%

#
old_mi = mi_total
mis = np.zeros(81)
X_deleting = X_val.clone()
X_mask = np.ones_like(X_val)
for i in range(81):
    # remove pixels with the worst mutual information one at a time
    mi = mi_one_pixel_removed(X_deleting, y_val)

    # get coordinates of the pixel that's worse to remove,
    # otherwise return the pixel with the highest entropy if there's a tie
    worst_pixel = (
        np.unravel_index(np.argmax((old_mi - mi) * X_mask), mi.shape)
        if (old_mi - mi).max() > 0
        else np.unravel_index(np.argmax(entropy(X_deleting.numpy())), mi.shape)
    )
    print(
        f"Step {i}, Worst pixel to remove: {worst_pixel}, MI diff: {old_mi - mi[worst_pixel]:.3f}"
    )
    old_mi = mi[worst_pixel]
    X_deleting[:, worst_pixel[0], worst_pixel[1]] = 0
    mis[i] = old_mi
    X_mask[:, worst_pixel[0], worst_pixel[1]] = 0

# %%
# plot mis
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(mis)
ax.set_title("Mutual information after greedily removing pixels")
ax.set_xlabel("Number of pixels removed")
ax.set_ylabel("Mutual information between image and label")
# %%

# repeat the above but consider MI with pixel 2, 3 instead
old_mi = mi_total
mis = np.zeros(81)
X_deleting = X_val.clone()
X_mask = np.ones_like(X_val)
index = np.unravel_index(np.argmax(entropy(X_val.numpy())), mi.shape)
X_mask[:, index[0], index[1]] = 0
X_deleting[:, index[0], index[1]] = 0
for i in range(1, 81):
    # remove pixels with the worst mutual information one at a time
    mi = mi_one_pixel_removed(X_deleting, X_val[:, index[0], index[1]])

    # get coordinates of the pixel that's worse to remove,
    # otherwise return the pixel with the highest entropy if there's a tie
    worst_pixel = (
        np.unravel_index(np.argmax((old_mi - mi) * X_mask), mi.shape)
        if (old_mi - mi).max() > 0
        else np.unravel_index(np.argmax(entropy(X_deleting.numpy())), mi.shape)
    )
    print(
        f"Step {i}, Worst pixel to remove: {worst_pixel}, MI diff: {old_mi - mi[worst_pixel]:.3f}"
    )
    old_mi = mi[worst_pixel]
    X_deleting[:, worst_pixel[0], worst_pixel[1]] = 0
    mis[i] = old_mi
    X_mask[:, worst_pixel[0], worst_pixel[1]] = 0
# %%
# plot mis
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(mis)
ax.set_title("Mutual information after greedily removing pixels")
ax.set_xlabel("Number of pixels removed")
ax.set_ylabel(f"Mutual information between image and X[{index[0]},{index[1]}]")

# %%
# train a small conv net on the MNIST dataset
# %%
# load the MNIST dataset
train_dataset = datasets.MNIST(
    root="data", train=True, transform=transforms.ToTensor(), download=True
)
test_dataset = datasets.MNIST(root="data", train=False, transform=transforms.ToTensor())
# %%
# create the data loaders
batch_size = 128
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)
# %%
# define the model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# %%
# initialize the model
model = ConvNet()
# %%
# define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))
# %%
# define the loss function
criterion = nn.CrossEntropyLoss()
# %%
# train the model
num_epochs = 5
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}"
            )
    # report test loss, accuracy
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(
            f"Test Accuracy of the model on the 10000 test images: {correct / total * 100:.2f} %"
        )

# %%
# get intermediate activations on valid set
def get_activations(model, loader):
    activations = []
    all_labels = []
    for i, (images, labels) in enumerate(loader):
        # forward pass
        x = F.relu(model.conv1(images))
        x = F.max_pool2d(x, 2)
        x = F.relu(model.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.reshape(x.size(0), -1)
        x = F.relu(model.fc1(x))
        activations.append(x)
        all_labels.append(labels)
    return torch.cat(activations), torch.cat(all_labels)


# %%
activations, labels = get_activations(model, train_loader)

# %%
# discretize the activations to 0, 1, 2, 3, 4
normalized_activations = activations / (activations.max(0)[0] + 1e-15)
discretized_activations = (
    torch.floor(normalized_activations * 5).type(torch.int64).detach().numpy()
)
# %%
# compute the mutual information between the discretized activations and the labels
def activations_to_ints(activations, nbins=5):
    return np.sum(activations * nbins ** np.arange(activations.shape[1]), axis=1)


# %%
int_activations = activations_to_ints(discretized_activations)
# %%
lookup_table = dict()
for act, label in zip(int_activations, labels):
    if act not in lookup_table:
        lookup_table[act] = torch.zeros(10)
    lookup_table[act][label] += 1

clean_lookup_table = dict()
for act, counts in lookup_table.items():
    label = torch.argmax(counts)
    clean_lookup_table[act] = label

# %%
# calculate accuracy of lookup table
correct = 0
for act, label in zip(int_activations, labels):
    if clean_lookup_table[act] == label:
        correct += 1
print(f"Accuracy of lookup table: {correct / len(labels) * 100:.2f} %")
# %%
# do a logistic regression on the activations to labels
from sklearn.linear_model import LogisticRegression  # type: ignore

lr = LogisticRegression()
lr.fit(discretized_activations, labels)
lr.score(discretized_activations, labels)
# %%
mutual_info_score(labels.numpy(), int_activations)
# %%
# `entropy`
def mi_one_feature_removed(X_val, y_val):
    mi = np.zeros(X_val.shape[-1])
    for i in range(X_val.shape[-1]):
        X_val_removed = X_val.copy()
        X_val_removed[:, i] = 0
        X_val_removed_int = activations_to_ints(X_val_removed)
        mi[i] = mutual_info_score(X_val_removed_int, y_val)

    return mi


# %%
old_mi = mi_total
mis = np.zeros(activations.shape[-1])
X_deleting = discretized_activations.copy()
X_mask = np.ones_like(X_deleting)
for i in range(activations.shape[-1]):
    # remove features one at a time to compute the mutual information
    mi = mi_one_feature_removed(X_deleting, labels.numpy())

    # get coordinates of the pixel that's worse to remove,
    # otherwise return the pixel with the highest entropy if there's a tie
    worst_pixel = (
        np.argmax((old_mi - mi) * X_mask)
        if (old_mi - mi).max() > 0
        else np.argmax(entropy(X_deleting))
    )
    print(
        f"Step {i}, Worst feature to remove: {worst_pixel}, MI diff: {old_mi - mi[worst_pixel]:.3f}"
    )
    old_mi = mi[worst_pixel]
    X_deleting[:, worst_pixel] = 0
    mis[i] = old_mi
    X_mask[:, worst_pixel] = 0

# %%
# plot the mutual information as we remove features
plt.plot(mis)
plt.xlabel("Number of features removed")
plt.ylabel("Mutual information")
plt.title("Mutual information remaining after removing features")
plt.show()

# %%
