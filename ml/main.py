from .edge_detection import (
    Peaks,
    find_horizontal_peaks,
    find_vertical_peaks,
    gaussian_blur,
    get_horizontal,
    get_image,
    pipeline,
    show,
    get_vertical,
    find_vertical_peaks,
)
from os import path
import os
import cv2
import torch
from .model_utils import load_model
from torchvision import transforms
import numpy as np
from string import ascii_letters


@torch.no_grad()
def get_model_result(filename, model_path):
    if not path.isfile(filename):
        exit(1)
    image = get_image(filename)
    vpeaks = pipeline(image, get_vertical, find_vertical_peaks)
    hpeaks = pipeline(image, get_horizontal, find_horizontal_peaks)

    peaks = Peaks(vpeaks, hpeaks)

    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    bgr_with_peaks = peaks.draw_peaks(bgr)
    # show(bgr, bgr_with_peaks)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize(48), transforms.CenterCrop(48)]
    )
    xs = sorted(vpeaks)
    ys = sorted(hpeaks)
    new_bgrs = []

    for i, _ in enumerate(xs[:-1]):
        for j, _ in enumerate(ys[:-1]):
            new_bgrs.append(transform(bgr[ys[j] : ys[j + 1], xs[i] : xs[i + 1]]))

    model = load_model(path.join(model_path, "model.pt"))
    new_bgrs = torch.stack(new_bgrs)

    translate = {
        "ee": " ",
        "bp": "󰡙",
        "bb": "󰡜",
        "bn": "󰡘",
        "bk": "󰡗",
        "bq": "󰡚",
        "br": "󰡛",
        "wp": "",
        "wb": "",
        "wn": "",
        "wk": "",
        "wq": "",
        "wr": "",
    }
    class_names = os.listdir(path.join(model_path, "data"))
    class_names.sort()  # Sort to maintain consistent class indexing
    ind_to_class = {idx: class_name for idx, class_name in enumerate(class_names)}

    out = model(new_bgrs)
    pred = out.argmax(dim=-1)
    classes = [ind_to_class[p.item()] for p in pred]
    classes_pretty = [translate[ind_to_class[p.item()]] for p in pred]

    classes = np.array(classes).reshape(8, 8).transpose()
    classes_pretty = np.array(classes_pretty).reshape(8, 8).transpose()

    for i, line in enumerate(classes_pretty):
        print(f"{8-i} ", end="")
        for char in line:
            print(f"[{char}]", end="")
        print()
    print("  ", end="")
    for letter in ascii_letters[:8]:
        print(f" {letter} ", end="")
    print()

    return classes
