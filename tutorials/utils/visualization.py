"""Visualization helpers for the quanda demo notebooks.

Shipped with the tutorials only — not part of the quanda library proper.
"""

from typing import Optional

import matplotlib.pyplot as plt
import torch
from matplotlib import font_manager, rcParams

_FONTS = ["../assets/demo/Helvetica.ttf", "../assets/demo/Helvetica-Bold.ttf"]
for _font in _FONTS:
    try:
        font_manager.fontManager.addfont(_font)
    except (FileNotFoundError, RuntimeError):
        pass

rcParams["font.family"] = "Helvetica"
rcParams["font.weight"] = "normal"


def _to_image(t: torch.Tensor) -> torch.Tensor:
    """Min-max normalize a CHW tensor for display."""
    t = t.detach().cpu().float()
    return (t - t.min()) / (t.max() - t.min() + 1e-8)


def visualize_top_bottom_influential(
    train_dataset,
    test_data: torch.Tensor,
    test_targets,
    predicted,
    influence_scores: torch.Tensor,
    label_dict: dict,
    top_k: int = 3,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Plot top-k proponents and opponents for each test sample.
    """
    n = len(test_data)
    cols = 1 + top_k + 1 + top_k  
    fig, axes = plt.subplots(
        n,
        cols,
        figsize=(1.6 * cols, 2.0 * n),
        gridspec_kw={"wspace": 0.1, "hspace": 0.6},
        squeeze=False,
    )

    top = torch.topk(influence_scores, top_k, dim=1, largest=True)
    bot = torch.topk(influence_scores, top_k, dim=1, largest=False)

    for row in range(n):
        true_label = label_dict.get(int(test_targets[row]), "?")
        pred_label = label_dict.get(int(predicted[row]), "?")
        correct = int(test_targets[row]) == int(predicted[row])
        pred_color = "#2ca02c" if correct else "#d62728"

        ax = axes[row, 0]
        ax.imshow(_to_image(test_data[row]).permute(1, 2, 0))
        ax.set_title(
            f"true: {true_label}\npred: {pred_label}",
            fontsize=10,
            color=pred_color,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(pred_color)
            spine.set_linewidth(2)

        for k in range(top_k):
            idx = int(top.indices[row, k])
            score = float(top.values[row, k])
            img, lbl = train_dataset[idx]
            ax = axes[row, 1 + k]
            ax.imshow(_to_image(img).permute(1, 2, 0))
            ax.set_title(
                f"{label_dict.get(int(lbl), '?')}\n{score:.3f}",
                fontsize=9,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0 and k == 0:
                ax.text(
                    -0.1,
                    1.45,
                    "proponents",
                    transform=ax.transAxes,
                    fontsize=11,
                    fontweight="bold",
                    color="#2ca02c",
                )

        axes[row, 1 + top_k].axis("off")

        for k in range(top_k):
            idx = int(bot.indices[row, k])
            score = float(bot.values[row, k])
            img, lbl = train_dataset[idx]
            ax = axes[row, 2 + top_k + k]
            ax.imshow(_to_image(img).permute(1, 2, 0))
            ax.set_title(
                f"{label_dict.get(int(lbl), '?')}\n{score:.3f}",
                fontsize=9,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0 and k == 0:
                ax.text(
                    -0.1,
                    1.45,
                    "opponents",
                    transform=ax.transAxes,
                    fontsize=11,
                    fontweight="bold",
                    color="#d62728",
                )

    if title:
        fig.suptitle(title, fontsize=14, y=1.0)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
