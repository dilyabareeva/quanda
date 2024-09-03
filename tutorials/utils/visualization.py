import matplotlib.pyplot as plt
import torch

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
