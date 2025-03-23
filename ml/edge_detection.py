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


def get_image(image_path: str):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    top, bottom, left, right = 1, 1, 1, 1
    grey_margin = 0
    return cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, None, value=grey_margin
    )


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
    plt.title("Original Image")
    plt.axis(axis)

    plt.subplot(1, 2, 2)
    plt.imshow(new, cmap="gray")
    plt.title("Edge Detection")
    plt.axis(axis)

    plt.show()


def find_vertical_peaks(convolved_image, show=False):
    depend_vertical = np.sum(convolved_image, 0)
    peaks = _find_peaks(depend_vertical)
    if show:
        plt.plot(depend_vertical)
        plt.scatter(peaks, depend_vertical[peaks], c="r")
        plt.show()
    return peaks


def find_horizontal_peaks(convolved_image, show=False):
    depend_horizontal = np.sum(convolved_image, 1)
    peaks = _find_peaks(depend_horizontal)
    if show:
        plt.plot(depend_horizontal)
        plt.scatter(peaks, depend_horizontal[peaks], c="r")
        plt.show()
    return peaks


def _find_peaks(depend):
    m = np.max(depend)
    inds = np.where(depend > m / 10)[0]
    sorted_indexes = np.argsort(depend)
    peaks = sorted_indexes[::-1][:50]
    peaks = sorted(list(set(inds) & set(peaks)))

    edges = 9
    metrics = {}
    for index, i in enumerate(peaks):
        for j in peaks[index + edges - 1 :]:
            metrics[(i, j)] = np.sum(depend[np.linspace(i, j, edges, dtype=int)])

    i, j = max(metrics, key=metrics.get)
    return np.linspace(i, j, edges, dtype=int).tolist()


def pipeline(input_, *functions):
    for function in functions:
        input_ = function(input_)
    return input_


def gaussian_blur(image):
    return cv2.GaussianBlur(image, (10, 10), 0)
