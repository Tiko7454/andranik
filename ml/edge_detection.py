import cv2
import numpy as np
import matplotlib.pyplot as plt
from copy import copy


class Peaks:
    def __init__(self, vpeaks, hpeaks):
        self.vpeaks = vpeaks
        self.hpeaks = hpeaks

    def draw_peaks(self, image):
        image = copy(image)
        for x in self.vpeaks:
            cv2.line(image, (x, 0), (x, image.shape[0]), (0, 0, 255), 1)
        for y in self.hpeaks:
            cv2.line(image, (0, y), (image.shape[1], y), (0, 0, 255), 1)
        return image


def get_image(image_path: str, scale):
    return cv2.imread(image_path, scale)


_KERNEL = np.array(
    [
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1],
    ]
)

KERNEL = _KERNEL / np.sum(abs(_KERNEL))


def get_horizontal(image, depth=-1):
    return _convolution(image, KERNEL.T, depth)


def get_vertical(image, depth=-1):
    return _convolution(image, KERNEL, depth)


def _convolution(image, kernel, depth):
    return abs(cv2.filter2D(image, depth, kernel)) + abs(
        cv2.filter2D(image, depth, -kernel)
    )


def show(old, new, axis="on"):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(old, cmap="gray")
    # plt.imshow(old)
    plt.title("Original Image")
    plt.axis(axis)

    plt.subplot(1, 2, 2)
    plt.imshow(new, cmap="gray")
    # plt.imshow(new)
    plt.title("Canny Edge Detection")
    plt.axis(axis)

    plt.show()


def find_vertical_peaks(convolved_image, show=True):
    depend_vertical = np.sum(convolved_image, 0)
    peaks = _find_peaks(depend_vertical)
    if show:
        plt.plot(depend_vertical)
        plt.scatter(peaks, depend_vertical[peaks], c="r")
        plt.show()
    return peaks


def find_horizontal_peaks(convolved_image, show=True):
    depend_horizontal = np.sum(convolved_image, 1)
    peaks = _find_peaks(depend_horizontal)
    if show:
        plt.plot(depend_horizontal)
        plt.scatter(peaks, depend_horizontal[peaks], c="r")
        plt.show()
    return peaks


def _find_peaks(depend, in_top=50):
    sorted_indexes = np.argsort(depend)

    peaks = []
    for i in range(in_top):
        x = sorted_indexes[-(i + 1)]

        if 0 < x < (depend.shape[0] - 1):
            if depend[x] >= depend[x + 1] and depend[x] >= depend[x - 1]:
                peaks.append(x)
    start, end = sorted(peaks[:2])
    return np.linspace(start, end, 9, dtype=int).tolist()


def pipeline(input_, *functions):
    for function in functions:
        input_ = function(input_)
    return input_


def gaussian_blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)
