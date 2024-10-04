import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv
from matplotlib import font_manager, rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Load the environment variables from .env file
load_dotenv()

# Set WandB API key
wandb_api_key = os.getenv("WANDB_API_KEY")
if not wandb_api_key:
    raise ValueError("WANDB_API_KEY not found in environment. Please add it to your .env file.")

# Authenticate with WandB
wandb.login(key=wandb_api_key)

# Define WandB project and filters
project_name = "quanda"
methods = ["representer_points", "trak", "random", "tracincpfast", "arnoldi"]
metrics = ["mislabeling_si", "shortcut", "subclass", "top_k_cardinality", "mixed_dataset"]

# Initialize a WandB API object
api = wandb.Api()

# Retrieve all runs from the project, sorted by creation time in descending order
runs = api.runs(f"{project_name}", order="+created_at")

all_results = []

# Initialize a dictionary to hold the latest results
results = {metric: {method: None for method in methods} for metric in metrics}

# Process each run and extract the latest values for each method and metric
for run in runs:
    # Get the creation date of the run
    created_at = run.created_at

    # Check for method-metric key in the run history
    for method in methods:
        for metric in metrics:
            key = f"{method}_{metric}"
            if key in run.summary:
                # Append to the results list
                all_results.append(
                    {"method": method, "metric": metric, "score": run.summary[key]["score"], "created_at": created_at}
                )

# Create a DataFrame from the results
df = pd.DataFrame(all_results)

# Remove duplicates based on method, metric, keeping the latest entry
df = df.loc[df.groupby(["method", "metric"])["created_at"].idxmax()]

# Pivot the DataFrame to have explainers as indices and metrics as columns
df = df.pivot(index="method", columns="metric", values="score")


# change values to rank, higher is better
df = df.rank(axis=0, method="max", ascending=True)

df.index.name = "explainer"

# Rename metrics
df = df.rename(
    columns={
        "top_k_cardinality": "Top-K\nCardinality",
        "subclass": "Subclass\nDetection",
        "mislabeling": "Mislabeling\nDetection",
        "shortcut": "Shortcut\nDetection",
        "random": "Model\nRandomisation",
        "mislabeling_si": "Mislabeling\nDetection",
        "mixed_dataset": "Mixed Dataset\nSeparation",
    }
)

metrics = df.columns.tolist()

# Rename methods
df = df.rename(
    index={
        "similarity": "Similarity Influence",
        "representer_points": "ReprPoints",
        "trak": "TRAK-1",
        "random": "Random",
        "tracincpfast": "TracInCP",
        "arnoldi": "ArnoldiInf",
    }
)

# sort indices by a list
sort_list = ["ReprPoints", "ArnoldiInf", "TracInCP", "TRAK-1", "Random"]

df = df.loc[sort_list]

# Optionally reset index to have a clean DataFrame
df.reset_index(inplace=True)

print(df.head())


fonts = ["../assets/demo/Helvetica.ttf", "../assets/demo/Helvetica-Bold.ttf"]
[font_manager.fontManager.addfont(font) for font in fonts]

# Set the font family to Helvetica
rcParams["font.family"] = "Helvetica"

# Optionally set the font weight and size if desired
rcParams["font.weight"] = "normal"  # or 'bold' if you want bold text
# rcParams['font.size'] = 10  # Set the default font size

BG_WHITE = "#ffffff"
BLUE = "#2a475e"
GREY70 = "#B0A8B9"
GREY_LIGHT = "#F4F4F4"  # "#f2efe8" #"#F8F3F3"

TRANSP_COLORS = ["#FFDDBB", "#83BA59", "#FFDAD4", "#EDDCFF"]
COLORS = ["#EB9C38", "#83BA59", "#EA4E38", "#7D53BA", "#90918B", "#EB9C38"]
RADIUS_RATIO = 1.2

# The three species of penguins
SPECIES = df["explainer"].tolist()
SPECIES_N = len(SPECIES)
# The four variables in the plot
VARIABLES = df.columns.tolist()[1:]
VARIABLES_N = len(VARIABLES)
"""
# The angles at which the values of the numeric variables are placed
ANGLES = [n / VARIABLES_N * 2 * np.pi for n in range(VARIABLES_N)]
ANGLES += ANGLES[:1]

TANGLES = copy.deepcopy(ANGLES)
# TANGLES[1] -= 0.05 * np.pi
# TANGLES[-2] += 0.05 * np.pi

# Padding used to customize the location of the tick labels
X_VERTICAL_TICK_PADDING = 5
X_HORIZONTAL_TICK_PADDING = 50

# Angle values going from 0 to 2*pi
HANGLES = np.linspace(0, 2 * np.pi)

# Used for the equivalent of horizontal lines in cartesian coordinates plots
# The last one is also used to add a fill which acts a background color.
H0 = np.zeros(len(HANGLES))
H1 = np.ones(len(HANGLES)) * 0.5
H2 = np.ones(len(HANGLES))
HS = [(n / (SPECIES_N - 1)) * np.ones(len(HANGLES)) for n in range(0, SPECIES_N)]

# Initialize layout ----------------------------------------------
width_pt = 310
height_pt = 186
fig = plt.figure(figsize=(width_pt / 72.27, height_pt / 72.27), dpi=300)
ax = fig.add_subplot(111, polar=True)


fig.patch.set_facecolor(BG_WHITE)
ax.set_facecolor(BG_WHITE)

# Rotate the "" 0 degrees on top.
# There it where the first variable, avg_bill_length, will go.
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Setting lower limit to negative value reduces overlap
# for values that are 0 (the minimums)
ax.set_ylim(-0.1, 1.05)

# Plot lines and dots --------------------------------------------
for idx, species in enumerate(SPECIES):
    values = (df.iloc[idx].drop("explainer").values / RADIUS_RATIO).tolist()
    values += values[:1]
    ax.plot(ANGLES, values, c=COLORS[idx], linewidth=1, label=species)
    ax.scatter(ANGLES, values, s=12, c=COLORS[idx], zorder=10)

# Set values for the angular axis (x)
ax.set_xticks(TANGLES[:-1])
ax.set_xticklabels(VARIABLES, size=8)
ax.tick_params(axis="x", pad=-8)

# Adjust individual tick label positions
for tick_label, angle in zip(ax.get_xticklabels(), TANGLES[:-1]):
    if tick_label.get_text() in ["Top-K\nCardinality", "Mixed Dataset\nSeparation"]:
        tick_label.set_y(tick_label.get_position()[1] - 0.1)
    if tick_label.get_text() == "Mislabeling\nDetection":
        tick_label.set_y(tick_label.get_position()[1] + 0.05)

# Remove lines for radial axis (y)
ax.set_yticks([])
ax.yaxis.grid(False)
ax.xaxis.grid(False)

# Remove spines
ax.spines["start"].set_color("none")
ax.spines["polar"].set_color("none")

# Add custom lines for radial axis (y) at 0, 0.5 and 1.
for j in range(len(HS)):
    ax.plot(HANGLES, HS[j] / RADIUS_RATIO, ls=(0, (2, 2)), c=GREY70, linewidth=0.65)

# Now fill the area of the circle with radius 1.
# This create the effect of gray background.
ax.fill(HANGLES, H2 / RADIUS_RATIO, GREY_LIGHT)

# Custom guides for angular axis (x).
# These four lines do not cross the y = 0 value, so they go from
# the innermost circle, to the outermost circle with radius 1.
for j in range(len(ANGLES)):
    ax.plot([ANGLES[j], ANGLES[j]], [0, 1 / RADIUS_RATIO], lw=0.5, c=GREY70)


handles = [
    Line2D(
        [0, 0.005],
        [0, 0],  # Short line segment
        color=color,  # Color for the line
        lw=1,  # Line width
        marker="o",  # Marker style
        markersize=3,  # Size of the dot
        markerfacecolor=color,  # Fill color for the marker
        label=species,  # Label for the legend
    )
    for species, color in zip(SPECIES, COLORS)
]

legend = ax.legend(
    handles=handles,
    loc=(1, 0),  # bottom-right
    labelspacing=0.15,  # add space between labels
    handlelength=1,  # set the length of the legend handle
    frameon=False,  # remove the frame
)

# Iterate through text elements and change their properties
for text in legend.get_texts():
    text.set_fontsize(8)  # Change default font size
    text.set_fontweight("bold")
# Set legend size by modifying the bounding box
# bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
# legend.set_bbox_to_anchor((0.05, 0.5, 0.2, 0.1))  # (x, y, width, height) in normalized figure coordinates

plt.tight_layout()
plt.show()

# save fig_1
fig.savefig("../scripts/fig_1.png", bbox_inches=None, pad_inches=0, dpi=1000)
"""

# Set figure size
width_pt = 170 / 72.27
height_pt = 110 / 72.27

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(width_pt, height_pt), dpi=500)
ax.set_facecolor("#FFFFFF")
fig.patch.set_facecolor("#FAFAF2")

# Number of groups (metrics)
n_metrics = len(metrics)
n_explainers = len(df)  # Number of explainers for each metric

# Bar width (slightly reduced for spacing)
bar_width = 0.12  # Reduce bar width for spacing between bars

# Space between bars within a group
bar_padding = 0.03  # Padding between bars

# X locations for groups (use metrics as x positions)
x_indices = np.arange(n_metrics)

# Plot horizontal dashed lines for y=1 to y=5
for y in [1, 2, 3, 4, 5][::-1]:
    plt.axhline(y=y, linewidth=0.3, zorder=0, color="gray", linestyle="dashed")

# Loop through each metric to plot bars
for j, metric in enumerate(metrics):
    # Get the values for this metric (one column of df)
    metric_values = df[metric].values

    # Sort values in descending order, and get the corresponding indices
    sorted_indices = np.argsort(metric_values)[::-1]
    sorted_values = metric_values[sorted_indices]

    # Adjust x positions for spacing between bars
    x_positions = x_indices[j] + np.arange(n_explainers) * (bar_width + bar_padding)

    # Plot bars for this metric (one set of bars)
    ax.bar(
        x_positions,
        sorted_values,
        width=bar_width,
        color=[COLORS[i % len(COLORS)] for i in sorted_indices],
        edgecolor="none",
        label=f"{metric}",
    )

# Set y-ticks to display 1, 2, 3, 4, 5
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_yticklabels([5, 4, 3, 2, 1])
ax.tick_params(axis="y", size=3, width=0.5)  # Reduce the y-tick marker size and width

# Set x-ticks with rotated metric names
ax.set_xticks(x_indices + (bar_width + bar_padding) * (n_explainers - 1) / 2)  # Center ticks
ax.set_xticklabels(metrics, rotation=45, ha="center", fontsize=7)  # Rotate metric names for better visibility
ax.tick_params(axis="x", pad=0, size=3, width=0.5)

# Remove the y-axis label
ax.set_ylabel("Rank", fontsize=7)  # Set y-axis label to empty

# set label font size

# Set y-tick label size to 6
ax.tick_params(axis="y", labelsize=6)

# Remove borders from the axes
for spine in ax.spines.values():
    spine.set_linewidth(0.3)
    spine.set_color("black")  # Ensure the spines are visible with color

plt.subplots_adjust(
    left=0.15, right=0.95, top=0.85, bottom=0.38
)  # Adjust the top margin (1 is the maximum, reduce for more space)


# Show plot
# plt.tight_layout()


plt.savefig("../scripts/bar_rank.png", bbox_inches=None, pad_inches=0, dpi=1000)
plt.show()
