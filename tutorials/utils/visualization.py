"""
The `visualization.py` file contains utility functions used in the tutorials to visualize the results
and is not part of the quanda library release. The code is not well-tested, well-documented and is provided
as a reference only.
"""

import matplotlib.pyplot as plt
import torch
from matplotlib import font_manager, rcParams
from torchvision.utils import save_image

fonts = ["../assets/demo/Poppins-Regular.ttf", "../assets/demo/Poppins-Bold.ttf"]
[font_manager.fontManager.addfont(font) for font in fonts]
rcParams["font.family"] = "Poppins"


def save_influential_samples(
    train_dataset, test_tensor, influence_scores, denormalize, test_names, r_name_dict, top_k=3, save_path="../assets/fig1"
):
    top_k_proponents = torch.topk(influence_scores, top_k, dim=1, largest=True)
    top_k_opponents = torch.topk(influence_scores, top_k, dim=1, largest=False)

    for idx, elements in enumerate(top_k_proponents.indices):
        proponents_images = [train_dataset[int(i)][0] for i in elements]
        proponent_labels = [r_name_dict[train_dataset[int(i)][1]] for i in elements]
        proponents_images = [denormalize(img) for img in proponents_images]
        for i, img in enumerate(proponents_images):
            label_i = proponent_labels[i]
            save_image(img, f"{save_path}/proponent_{idx}_{label_i}_top_{i}.png")

    for idx, elements in enumerate(top_k_opponents.indices):
        opponents_images = [train_dataset[int(i)][0] for i in elements]
        opponents_labels = [r_name_dict[train_dataset[int(i)][1]] for i in elements]
        opponents_images = [denormalize(img) for img in opponents_images]
        for i, img in enumerate(opponents_images):
            label_i = opponents_labels[i]
            save_image(img, f"{save_path}/opponent_{idx}_{label_i}_top_{i}.png")

    test_images = [denormalize(img) for img in test_tensor]
    for img, idx in zip(test_images, range(len(test_tensor))):
        save_image(img, f"{save_path}/test_{idx}_{test_names[idx]}.png")


#### Visualizuation code for explaining test samples


# %%
def visualize_influential_samples(train_dataset, test_tensor, influence_scores, top_k=3):
    top_k_proponents = torch.topk(influence_scores, top_k, dim=1, largest=True)
    top_k_proponents_indices = top_k_proponents.indices
    top_k_proponents_scores = top_k_proponents.values

    top_k_opponents = torch.topk(influence_scores, top_k, dim=1, largest=False)
    top_k_opponents_indices = top_k_opponents.indices
    top_k_opponents_scores = top_k_opponents.values

    def plot_samples(test_idx):
        proponents_images = [train_dataset[int(idx)][0] for idx in top_k_proponents_indices[test_idx]]
        proponents_images = [(img - img.min()) / (img.max() - img.min()) for img in proponents_images]
        proponents_scores = top_k_proponents_scores[test_idx]

        opponents_images = [train_dataset[int(idx)][0] for idx in reversed(top_k_opponents_indices[test_idx])]
        opponents_images = [(img - img.min()) / (img.max() - img.min()) for img in opponents_images]
        opponents_scores = list(reversed(top_k_opponents_scores[test_idx]))

        test_image = test_tensor[test_idx]
        test_image = (test_image - test_image.min()) / (test_image.max() - test_image.min())

        plt.figure(figsize=(24, 8))

        for i, (img, score) in enumerate(zip(opponents_images, opponents_scores)):
            plt.subplot(1, 7, i + 1)
            plt.imshow(img.permute(1, 2, 0))
            plt.gca().add_patch(
                plt.Rectangle((0, 0), img.shape[1], img.shape[2], linewidth=15, edgecolor="red", facecolor="none")
            )
            plt.title(f"Opponent {3 - i}\nScore: {score:.4f}")
            plt.axis("off")

        plt.subplot(1, 7, 4)
        plt.imshow(test_image.permute(1, 2, 0))
        plt.gca().add_patch(
            plt.Rectangle((0, 0), img.shape[1], img.shape[2], linewidth=15, edgecolor="green", facecolor="none")
        )
        plt.title(f"Test Sample {test_idx + 1}")
        plt.axis("off")

        for i, (img, score) in enumerate(zip(proponents_images, proponents_scores)):
            plt.subplot(1, 7, i + 5)
            plt.imshow(img.permute(1, 2, 0))
            plt.gca().add_patch(
                plt.Rectangle((0, 0), img.shape[1], img.shape[2], linewidth=15, edgecolor="blue", facecolor="none")
            )
            plt.title(f"Proponent {i + 1}\nScore: {score:.4f}")
            plt.axis("off")

        plt.show()

    for test_idx in range(len(test_tensor)):
        plot_samples(test_idx)


# %% md
#### Visualizuation code for self-influence scores
# %%
def visualize_self_influence_samples(train_dataset, self_influence_scores, top_k=5):
    top_k_most_influential = torch.topk(self_influence_scores, top_k, largest=True)
    top_k_least_influential = torch.topk(self_influence_scores, top_k, largest=False)

    top_k_most_indices = top_k_most_influential.indices
    top_k_most_scores = top_k_most_influential.values

    top_k_least_indices = top_k_least_influential.indices
    top_k_least_scores = top_k_least_influential.values

    def plot_samples():
        most_influential_images = [train_dataset[int(idx)][0] for idx in top_k_most_indices]
        most_influential_images = [(img - img.min()) / (img.max() - img.min()) for img in most_influential_images]

        least_influential_images = [train_dataset[int(idx)][0] for idx in top_k_least_indices]
        least_influential_images = [(img - img.min()) / (img.max() - img.min()) for img in least_influential_images]

        plt.figure(figsize=(20, 10))

        plt.subplot(2, top_k, 1)
        plt.text(
            -0.1, 0.5, "Most Influential", fontsize=16, ha="center", va="center", rotation=90, transform=plt.gca().transAxes
        )

        for i, (img, score) in enumerate(zip(most_influential_images, top_k_most_scores)):
            plt.subplot(2, top_k, i + 1)
            plt.imshow(img.permute(1, 2, 0))
            plt.gca().add_patch(
                plt.Rectangle((0, 0), img.shape[1], img.shape[2], linewidth=10, edgecolor="blue", facecolor="none")
            )
            plt.title(f"Score: {score:.2f}")
            plt.axis("off")

        plt.subplot(2, top_k, top_k + 1)
        plt.text(
            -0.1, 0.5, "Least Influential", fontsize=16, ha="center", va="center", rotation=90, transform=plt.gca().transAxes
        )

        # Plot the least influential samples
        for i, (img, score) in enumerate(zip(least_influential_images, top_k_least_scores)):
            plt.subplot(2, top_k, top_k + i + 1)
            plt.imshow(img.permute(1, 2, 0))
            plt.gca().add_patch(
                plt.Rectangle((0, 0), img.shape[1], img.shape[2], linewidth=10, edgecolor="red", facecolor="none")
            )
            plt.title(f"Score: {score:.2f}")
            plt.axis("off")

        plt.subplots_adjust(wspace=0.4)
        plt.tight_layout()
        plt.show()

    plot_samples()


def visualize_samples(images, labels, row_headers, denormalize, label_to_name_dict):
    if len(row_headers) != 4:
        raise ValueError("row_headers must have 4 elements")

    grid_size = (4, 3)
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(4, 4), dpi=180)

    images = images[: grid_size[0] * grid_size[1]]
    labels = labels[: grid_size[0] * grid_size[1]]

    for i, ax in enumerate(axes.flat):
        img = denormalize(images[i]).permute(1, 2, 0).numpy()
        label = label_to_name_dict[labels[i].item()]

        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"{label}", fontsize=8.5, color="black", fontweight="regular")

    # Add row descriptions to the left of each row (horizontally)
    for i, header in enumerate(row_headers):
        fig.text(0.02, 0.9 - (i / grid_size[0]), header, va="top", ha="right", fontsize=10, color="black", fontweight="bold")

    # Adjust spacing between images for even spacing
    plt.subplots_adjust(wspace=0.4, hspace=0.4, left=0.1, right=0.95)
    plt.tight_layout()  # Adjusted the rect parameter for better alignment
    plt.show()
