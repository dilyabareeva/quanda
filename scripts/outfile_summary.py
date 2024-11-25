import os

import torch

dir = "/home/fe/yolcu/Documents/Code/quanda/test_output"
outfile = "/home/fe/yolcu/Documents/Code/quanda/test_output/outfile"
names = []
vals = []
trains = []
for f in os.listdir(dir):
    if not f.endswith(".out"):
        continue
    with open(os.path.join(dir, f)) as file:
        lines = file.readlines()
    temp_vals = []
    temp_trains = []
    final_train = None
    for l in lines:
        if "vanilla" in l or "mislabeled" in l or "shortcut" in l or "subclass" in l or "mixed" in l:
            name = l.replace("\n", "")
        if "validation" in l:
            temp_vals.append(float(l.split(" ")[-1]))
            temp_trains.append(final_train)
        if "train" in l:
            final_train = float(l.split(" ")[-1])
    temp_vals = torch.tensor(temp_vals)
    temp_trains = torch.tensor(temp_trains)
    vals.append(temp_vals.max())
    trains.append(temp_trains[temp_vals.argmax()])
    names.append(name)

vals = torch.tensor(vals)
trains = torch.tensor(trains)
indices = vals.argsort()
with open(os.path.join(dir, outfile), "w") as o:
    for i in range(len(indices)):
        o.write(names[indices[i]])
        print(names[indices[i]])
        o.write(f"val: {vals[indices[i]]}")
        print(f"val: {vals[indices[i]]}")
        o.write(f"train: {trains[indices[i]]}")
        print(f"train: {trains[indices[i]]}")
        o.write("\n\n*****\n\n")
        print("\n\n*****\n\n", end="")
