from downscale_canny import (
    find_horizontal_peaks,
    find_vertical_peaks,
    get_horizontal,
    get_image,
    show,
    get_vertical,
    find_vertical_peaks,
    draw_vertical_peaks,
    draw_horizontal_peaks,
)
import cv2


def main():
    image = get_image("test.png", cv2.IMREAD_GRAYSCALE)
    imv = cv2.GaussianBlur(get_vertical(image), (5, 5), 0)
    imh = cv2.GaussianBlur(get_horizontal(image), (5, 5), 0)
    vpeaks = find_vertical_peaks(imv)
    hpeaks = find_horizontal_peaks(imh)

    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    img = draw_vertical_peaks(bgr, vpeaks)
    img = draw_horizontal_peaks(img, hpeaks)
    show(imv, img)


if __name__ == "__main__":
    main()
