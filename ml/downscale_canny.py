import cv2
import matplotlib.pyplot as plt


def downscale_and_canny(image, d=128, t1=100, t2=200):
    image = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (d, d), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(image, t1, t2)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap="gray")
    plt.title("Canny Edge Detection")
    plt.axis("off")

    plt.show()
