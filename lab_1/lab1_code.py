from skimage.color import rgb2gray
from skimage.io import imread
import skimage.transform as tf
import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure
plt.gray()

lena= rgb2gray(imread('lena.tiff')) *255
cameraman = imread('cameraman.tif').astype(np.float64)
tire = imread('tire.tif').astype(np.float64) / 255.0

method_order_to_name = {
    0: "Nearest Neighbor",
    1: "Bilinear",
    3: "Bicubic"
}

def mse(f, g):
    """
    Mean squared error between 2 images
    """
    m, n = f.shape
    return np.sum(np.square(f - g)) * (1 / (m*n))

def PSNR(f, g):
    """
    Peak signal to noise ratio between 2 images
    """
    f = f.astype(np.float64)
    g = g.astype(np.float64)

    return 10 * np.log10(255.0**2 / mse(f, g))

def reduce_resolution(image, title, factor):
    """
    reduce each image resolution by a factor of 4 (horizontally and vertically)
    """
    new_shape = image.shape[0] // factor, image.shape[1] // factor
    print(f"Old shape: {image.shape}, New shape: {new_shape}")
    image = tf.resize(image, new_shape, order=1)
    plt.imshow(image)
    plt.title(title)
    plt.show()
    print(image.shape)
    return image

def digital_zoom(image, factor, order, ax, title):
    """
    Perform digital zoom where order NN = 0, bilinear = 1, bicubic = 3
    """
    new_shape = image.shape[0] * factor, image.shape[1] * factor
    print(f"Old shape: {image.shape}, New shape: {new_shape}")
    image = tf.resize(image, new_shape, order=order)
    ax.imshow(image)
    ax.title.set_text(title)
    return image

def zoom_image_different_methods(image, factor):
    images = []
    fig, axs = plt.subplots(1, 3, figsize=(15, 15))
    for i in range(3):
        order = i
        if i == 2:
            order = 3
        zoomed_image = digital_zoom(image, factor, order, axs[i], f"Digital Zooming with {method_order_to_name[order]} Interpolation")
        images.append(zoomed_image)
    return images


# plot original images
fig, axs = plt.subplots(1, 2, figsize=(8, 8))
for i, image in enumerate([lena, cameraman]):
    axs[i].imshow(image, cmap='gray')

fig.suptitle('Lena and Cameraman Original Images')
plt.subplots_adjust(top=1.4)
plt.show()

# reduce resolution and plot
lena_reduced = reduce_resolution(lena, "Lena Reduced Resolution", 4)
cameraman_reduced = reduce_resolution(cameraman, "Cameraman Reduced Resolution", 4)

# zoom images and plot
lena_images = zoom_image_different_methods(lena_reduced, 4)
plt.show()
cameraman_images = zoom_image_different_methods(cameraman_reduced, 4)
plt.show()

# print the PSNR values for Lena and the cameraman
for i, l_image in enumerate(lena_images):
    order = i
    if i == 2:
        order = 3
    upsample_method = method_order_to_name[order]
    print(f"PSNR between Lena image and up-sampled image using {upsample_method.lower()} interpolation: {PSNR(lena, l_image)}")


for i, c_image in enumerate(cameraman_images):
    order = i
    if i == 2:
        order = 3
    upsample_method = method_order_to_name[order]
    print(f"PSNR between Cameraman image and up-sampled image using {upsample_method.lower()} interpolation: {PSNR(cameraman, c_image)}")
