from downscale_canny import (
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
import cv2


def main():
    image = get_image("test.png", cv2.IMREAD_GRAYSCALE)
    vpeaks = pipeline(image, get_vertical, gaussian_blur, find_vertical_peaks)
    hpeaks = pipeline(image, get_horizontal, gaussian_blur, find_horizontal_peaks)
    peaks = Peaks(vpeaks, hpeaks)

    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    bgr_with_peaks = peaks.draw_peaks(bgr)
    show(bgr, bgr_with_peaks)


if __name__ == "__main__":
    main()
