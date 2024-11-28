"""The `visualization.py` file contains utility functions used in the tutorials to visualize the results
and is not part of the quanda library release. The code is not well-tested, well-documented and is provided
as a reference only.
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import torch
from matplotlib import font_manager, rcParams
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from torchvision.utils import save_image

fonts = ["../assets/demo/Helvetica.ttf", "../assets/demo/Helvetica-Bold.ttf"]
[font_manager.fontManager.addfont(font) for font in fonts]

# Set the font family to Helvetica
rcParams["font.family"] = "Helvetica"

# Optionally set the font weight and size if desired
rcParams["font.weight"] = "normal"  # or 'bold' if you want bold text
# rcParams['font.size'] = 10  # Set the default font size


def save_influential_samples(
    train_dataset,
    test_tensor,
    influence_scores,
    denormalize,
    test_names,
    r_name_dict,
    top_k=3,
    save_path="../assets/fig1",
    method="repr_points",
    color="#1f77b4",
):
    top_k = 3

    # normalize influence scores by row
    influence_scores = (
        influence_scores - influence_scores.min(dim=1, keepdim=True)[0]
    )
    influence_scores = (
        influence_scores / influence_scores.max(dim=1, keepdim=True)[0]
    )

    # Get the top opponents and proponents
    top_k_proponents = torch.topk(influence_scores, top_k, dim=1, largest=True)
    top_k_opponents = torch.topk(influence_scores, top_k, dim=1, largest=False)

    proponents_images = []
    opponents_images = []
    proponent_labels = []
    opponent_labels = []

    # Collect proponents images
    for idx, elements in enumerate(top_k_proponents.indices):
        proponents_images.extend(
            [
                denormalize(train_dataset[int(i)][0]).permute(1, 2, 0).numpy()
                for i in elements
            ]
        )
        proponent_labels.extend(
            [
                top_k_proponents.values[idx][i].item()
                for i in range(len(elements))
            ]
        )

    # Collect opponents images
    for idx, elements in enumerate(top_k_opponents.indices):
        opponents_images.extend(
            [
                denormalize(train_dataset[int(i)][0]).permute(1, 2, 0).numpy()
                for i in elements
            ]
        )
        opponent_labels.extend(
            [
                top_k_opponents.values[idx][i].item()
                for i in range(len(elements))
            ]
        )

    # Reverse the order of proponents images
    proponents_images = proponents_images[::-1]
    proponent_labels = proponent_labels[::-1]

    # Set the desired width and calculate the total figure size
    image_width_pixels = 145
    dpi = 72.27  # Dots per inch
    height_in_inches = 0.32  # Keep height as needed

    # Create figure and axes for the images
    fig, axes = plt.subplots(
        nrows=1,
        ncols=top_k * 2 + 1,
        figsize=(image_width_pixels / dpi, height_in_inches),
        gridspec_kw={"hspace": 0.0, "wspace": 0.03},
        dpi=1000,
    )  # Reduced wspace to a smaller value

    # Set the figure background color
    fig.patch.set_facecolor(color)

    # Plot opponents images and labels
    for ax, img, label in zip(axes[:top_k], opponents_images, opponent_labels):
        ax.imshow(img)
        ax.axis("off")  # Hide axes
        # Set the background color for each image area
        ax.set_facecolor(color)  # Set background color for the image frame
        ax.text(
            0.5,
            -0.02,
            f"{label:.2f}",
            ha="center",
            va="top",
            fontsize=4,
            transform=ax.transAxes,
        )

    # Create empty axes for the space between proponents and opponents
    empty_ax = axes[top_k]  # Position the empty ax in between
    empty_ax.axis("off")  # Hide the empty axes

    # Plot proponents images and labels
    for ax, img, label in zip(
        axes[top_k + 1 :], proponents_images, proponent_labels
    ):
        ax.imshow(img)
        ax.axis("off")  # Hide axes
        # Set the background color for each image area
        ax.set_facecolor(color)  # Set background color for the image frame
        ax.text(
            0.5,
            -0.02,
            f"{label:.2f}",
            ha="center",
            va="top",
            fontsize=4,
            transform=ax.transAxes,
        )

    # Adjust layout to remove whitespace and set spacing
    plt.subplots_adjust(
        left=0, right=1, top=1.27, bottom=0, hspace=0.0, wspace=0.1
    )  # Set wspace to a very small value for tighter spacing

    # Save the figure if needed
    plt.savefig(
        f"{save_path}/test_{method}.png", pad_inches=0, dpi=1000
    )  # Save the plot without extra space

    plt.show()  # Show the plot

    # save test image with torch
    test_image = denormalize(test_tensor[0])
    save_image(test_image, f"{save_path}/test_image.png")


def visualize_top_3_bottom_3_influential(
    train_dataset,
    test_tensor,
    test_targets,
    predicted,
    influence_scores,
    label_dict,
    save_path=None,
):
    num_samples = len(test_tensor)
    plt.figure(figsize=(24, 5 * num_samples))
    gs = GridSpec(num_samples, 17, height_ratios=[1] * num_samples)

    all_colors = [
        color
        for color in mcolors.CSS4_COLORS.values()
        if mcolors.rgb_to_hsv(mcolors.to_rgb(color))[2] < 0.7
    ]

    unique_labels = list(label_dict.values())
    label_colors = {
        label: all_colors[i % len(all_colors)]
        for i, label in enumerate(unique_labels)
    }

    top_k = 3
    top_k_proponents = torch.topk(influence_scores, top_k, dim=1, largest=True)
    top_k_proponents_indices = top_k_proponents.indices
    top_k_proponents_scores = top_k_proponents.values

    top_k_opponents = torch.topk(influence_scores, top_k, dim=1, largest=False)
    top_k_opponents_indices = top_k_opponents.indices
    top_k_opponents_scores = top_k_opponents.values

    predicted_labels_str = [
        label_dict.get(int(label_num), "Unknown") for label_num in predicted
    ]

    for test_idx in range(num_samples):
        test_image = test_tensor[test_idx]
        test_image = (test_image - test_image.min()) / (
            test_image.max() - test_image.min()
        )
        test_label_num = test_targets[test_idx].item()
        test_label_str = label_dict.get(test_label_num, "Unknown")
        test_label_color = label_colors.get(test_label_str, "gray")

        proponents_data = [
            (train_dataset[int(idx)][0], train_dataset[int(idx)][1])
            for idx in top_k_proponents_indices[test_idx]
        ]
        proponents_images = [img for img, _ in proponents_data]
        proponents_images = [
            (img - img.min()) / (img.max() - img.min())
            for img in proponents_images
        ]
        proponents_labels_str = [
            label_dict.get(label_num, "Unknown")
            for _, label_num in proponents_data
        ]
        proponents_scores = top_k_proponents_scores[test_idx]

        opponents_data = [
            (train_dataset[int(idx)][0], train_dataset[int(idx)][1])
            for idx in reversed(top_k_opponents_indices[test_idx])
        ]
        opponents_images = [img for img, _ in opponents_data]
        opponents_images = [
            (img - img.min()) / (img.max() - img.min())
            for img in opponents_images
        ]
        opponents_labels_str = [
            label_dict.get(label_num, "Unknown")
            for _, label_num in opponents_data
        ]
        opponents_scores = list(reversed(top_k_opponents_scores[test_idx]))

        # Plot test sample
        ax = plt.subplot(gs[test_idx, 0:3])
        ax.imshow(test_image.permute(1, 2, 0))
        ax.text(
            0.0,
            1.0,
            f"{test_label_str}",
            transform=ax.transAxes,
            backgroundcolor=test_label_color,
            color="white",
            fontsize=16,
            verticalalignment="top",
            bbox=dict(facecolor=test_label_color, edgecolor="none", pad=10),
        )
        if test_label_str == predicted_labels_str[test_idx]:
            pred_color = "green"
        else:
            pred_color = "red"
        ax.add_patch(
            Rectangle(
                (0, -0.15),
                1,
                0.22,
                transform=ax.transAxes,
                color=pred_color,
                clip_on=False,
            )
        )
        ax.text(
            0.5,
            0.05,
            f"Predicted:\n{predicted_labels_str[test_idx]}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=16,
            color="white",
        )
        plt.axis("off")

        # Plot proponents
        for i, (img, score, label_str) in enumerate(
            zip(proponents_images, proponents_scores, proponents_labels_str)
        ):
            ax = plt.subplot(gs[test_idx, 4 + i * 2 : 4 + (i * 2 + 2)])
            label_color = label_colors.get(label_str, "gray")
            ax.imshow(img.permute(1, 2, 0))
            ax.text(
                0.0,
                1.0,
                f"{label_str}",
                transform=ax.transAxes,
                backgroundcolor=label_color,
                color="white",
                fontsize=16,
                verticalalignment="top",
                bbox=dict(facecolor=label_color, edgecolor="none", pad=8),
            )
            ax.add_patch(
                Rectangle(
                    (0, -0.15),
                    1,
                    0.12,
                    transform=ax.transAxes,
                    color="black",
                    clip_on=False,
                )
            )
            ax.text(
                0.5,
                -0.05,
                f"Score: {score:.4f}",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=16,
                color="white",
            )
            plt.axis("off")

        # Plot opponents
        for i, (img, score, label_str) in enumerate(
            zip(opponents_images, opponents_scores, opponents_labels_str)
        ):
            ax = plt.subplot(gs[test_idx, 11 + i * 2 : 11 + (i * 2 + 2)])
            label_color = label_colors.get(label_str, "gray")
            ax.imshow(img.permute(1, 2, 0))
            ax.text(
                0.0,
                1.0,
                f"{label_str}",
                transform=ax.transAxes,
                backgroundcolor=label_color,
                color="white",
                fontsize=16,
                verticalalignment="top",
                bbox=dict(facecolor=label_color, edgecolor="none", pad=8),
            )
            ax.add_patch(
                Rectangle(
                    (0, -0.15),
                    1,
                    0.12,
                    transform=ax.transAxes,
                    color="black",
                    clip_on=False,
                )
            )
            ax.text(
                0.5,
                -0.05,
                f"Score: {score:.4f}",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=16,
                color="white",
            )
            plt.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def visualize_samples(
    images, labels, row_headers, denormalize, label_to_name_dict
):
    if len(row_headers) != 4:
        raise ValueError("row_headers must have 4 elements")

    grid_size = (4, 3)
    fig, axes = plt.subplots(
        grid_size[0], grid_size[1], figsize=(6, 5.5), dpi=180
    )

    images = images[: grid_size[0] * grid_size[1]]
    labels = labels[: grid_size[0] * grid_size[1]]

    for i, ax in enumerate(axes.flat):
        img = denormalize(images[i]).permute(1, 2, 0).numpy()
        label = label_to_name_dict[labels[i].item()]

        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(
            f"{label}", fontsize=12, color="black", fontweight="regular"
        )

    # Add row descriptions to the left of each row (horizontally)
    for i, header in enumerate(row_headers):
        fig.text(
            0.02,
            0.9 - (i / grid_size[0]),
            header,
            va="top",
            ha="right",
            fontsize=14,
            color="black",
        )

    # Adjust spacing between images for even spacing
    plt.subplots_adjust(wspace=0.4, hspace=0.4, left=0.1, right=0.9)
    plt.tight_layout()  # Adjusted the rect parameter for better alignment
    plt.show()
