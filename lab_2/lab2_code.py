import scipy.signal as signal
import matplotlib.pyplot as plt
import skimage.util as sk_util
import scipy.ndimage as ndimage
from skimage.color import rgb2gray
from skimage.io import imread
import numpy as np
import matplotlib



plt.gray()
lena= rgb2gray(imread('../lab_images/lena.tiff'))
cameraman = imread('../lab_images/cameraman.tif').astype(np.float64)/255


def gaussian_filter(n_rows, n_cols, stdv):
    """
    Returns a 2d Gaussian image filter.
    """
    g_r = signal.windows.gaussian(n_rows, stdv)
    g_c = signal.windows.gaussian(n_cols, stdv)

    G = np.outer(g_r, g_c)

    return G/np.sum(G)

def imnoise_speckle(im, v):
    # im: input image
    # v: variance
    n = np.sqrt(v*12) * (np.random.rand(im.shape[0], im.shape[1]) - 0.5)
    return im + im * n

def PSNR(f,g):
    return 10*np.log10(1.0/ np.mean(np.square(f-g)))

def im_hist_show(image, image_title, histogram_title, vmin=0, vmax=1):
    """
    Plot both an image and associated histogram
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    image_flattened = image.flatten()
    axs[0].imshow(image, vmin=vmin, vmax=vmax)
    axs[1].hist(image_flattened)
    axs[0].set_title(image_title)
    axs[1].set_title(histogram_title)
    axs[1].set_xlabel("Pixel Intensity (k)")
    axs[1].set_ylabel("Pixel Count (n_k)")
    plt.show()

def avg_filter(n):
    avg_filter = np.ones((n,n))/(n*n)
    return avg_filter


#  section 2 of the lab
# create the impulse responses as filters
h1 = (1/6)*np.ones((1,6))
h2 = h1.T
h3 = np.array([[-1, 1]])
y1_lena = signal.convolve2d(lena, h1, mode="same")
y2_lena = signal.convolve2d(lena, h2, mode="same")
y3_lena = signal.convolve2d(lena, h3, mode="same")

plt.imshow(y1_lena)
plt.title("Lena Convolved With h1")
plt.show()

plt.imshow(y2_lena)
plt.title("Lena Convolved With h2")
plt.show()

plt.imshow(y3_lena, vmin=0, vmax=1)
plt.title("Lena Convolved With h3")
plt.show()

# section 3 of lab
f = np.hstack([0.3*np.ones((200,100)), 0.7*np.ones((200,100))])
im_hist_show(f, "Original Image", "Toy Image Histogram")

# apply zero mean and variance 0.01 gaussian noise
f_gaussian_noise = sk_util.random_noise(f, mode="gaussian")
im_hist_show(f_gaussian_noise, "Image with Gaussian Noise", "Histogram with Additive Gaussian Noise")

# apply salt and pepper noise
f_s_and_p_noise = sk_util.random_noise(f, mode="s&p")
im_hist_show(f_s_and_p_noise, "Image with Salt and Pepper Noise", "Histogram with Salt and Pepper Noise")

speckle_variance = 0.04
f_speckle_noise = imnoise_speckle(f, speckle_variance)
im_hist_show(f_speckle_noise, "Image with Speckle Noise", "Histogram with Speckle Noise")

# section 4 of lab
im_hist_show(lena,'Original Lena Image', 'Original Lena Histogram')

lena_gaussian = sk_util.random_noise(lena, mode='gaussian', var = 0.002)
im_hist_show(lena_gaussian,'Additive Gaussian Noise Lena Image', 'Additive Gaussian Noise Lena Histogram')
print(f"The PSNR between the original and gaussian noise lena image is: {round(PSNR(lena, lena_gaussian), 3)}")

# Plot the 3x3 averaging filter as a matrix
plt.figure(figsize=(6, 6))
plt.imshow(avg_filter(3), cmap='gray', interpolation='nearest')
plt.title('3x3 Averaging Filter')
plt.colorbar(label='Filter Value')
plt.xticks(np.arange(3), np.arange(1, 4))
plt.yticks(np.arange(3), np.arange(1, 4))
plt.grid(False)
plt.show()

denoised_lena_3x3 = ndimage.convolve(lena_gaussian, avg_filter(3))

im_hist_show(denoised_lena_3x3,'Denoised Lena Image with Average Filter 3x3', 'Denoised Lena Histogram with Average Filter 3x3')
print(f"The PSNR between the original and denoised lena image with the 3x3 average filter (with gaussian noise) is: {round(PSNR(denoised_lena_3x3, lena), 3)}")

avg_filter_7x7 = avg_filter(7)
denoised_lena_avg_7x7 = ndimage.convolve(lena_gaussian, avg_filter_7x7)
im_hist_show(denoised_lena_avg_7x7,'Denoised Lena Image with Average Filter 7x7', 'Denoised Lena Histogram with Average Filter 7x7')

print(f"The PSNR between the original and 7x7 average filter denoised lena image (with gaussian noise) is: {round(PSNR(denoised_lena_avg_7x7, lena), 3)}")

gaussian_filter_7x7 = gaussian_filter(7, 7, 1)

# Plot the 3x3 averaging filter as a matrix
plt.figure(figsize=(6, 6))
plt.imshow(gaussian_filter_7x7, cmap='gray', interpolation='nearest')
plt.title('7x7 Gaussian Filter')
plt.colorbar(label='Filter Value')
plt.xticks(np.arange(7), np.arange(1, 8))
plt.yticks(np.arange(7), np.arange(1, 8))
plt.grid(False)
plt.show()

denoised_lena_gaussian_7x7 = ndimage.convolve(lena_gaussian, gaussian_filter_7x7)
im_hist_show(denoised_lena_gaussian_7x7,'Denoised Lena Image with Gaussian Filter 7x7', 'Denoised Lena Histogram with Gaussian Filter 7x7')
print(f"The PSNR between the original and lena denoised with the gaussian filter (with gaussian noise) is: {round(PSNR(denoised_lena_gaussian_7x7, lena), 3)}")

lena_salt_pepper = sk_util.random_noise(lena, mode='s&p')
im_hist_show(lena_salt_pepper,'Salt & Pepper Noice Lena Image', 'Salt & Pepper Noice Lena Histogram')

denoised_lena_sp_avg_7x7 = ndimage.convolve(lena_salt_pepper, avg_filter(7))
im_hist_show(denoised_lena_sp_avg_7x7,'Denoised S&P Lena Image with Average Filter 7x7', 'Denoised S&P Lena Histogram with Average Filter 7x7')
print(f"The PSNR between the original and (7x7) average filter denoised lena image (with S&P noise) is: {round(PSNR(denoised_lena_sp_avg_7x7, lena), 3)}")

denoised_lena_sp_gaussian_7x7 = ndimage.convolve(lena_salt_pepper, gaussian_filter_7x7)
im_hist_show(denoised_lena_sp_gaussian_7x7,'Denoised S&P Lena Image with Gaussian Filter 7x7', 'Denoised S&P Lena Histogram with Gaussian Filter 7x7')
print(f"The PSNR between the original and gaussian filter denoised lena image (with S&P noise) is: {round(PSNR(denoised_lena_sp_gaussian_7x7, lena), 3)}")

denoised_lena_median_filter_3x3 = ndimage.median_filter(lena_salt_pepper, size=(3,3))
im_hist_show(denoised_lena_median_filter_3x3,'Denoised S&P Lena Image with Median Filter 3x3', 'Denoised S&P Lena Histogram with Median Filter 3x3')
print(f"The PSNR between the original and median filter smoothed lena image (with S&P noise) is: {round(PSNR(denoised_lena_median_filter_3x3, lena), 3)}")

# section 5 of lab

plt.imshow(cameraman)
plt.title("Original Cameraman Image")
plt.show()

cameraman_gaussian_filtered = ndimage.convolve(cameraman, gaussian_filter_7x7)
cameraman_minus_gaussian_filtered = cameraman - cameraman_gaussian_filtered
plt.imshow(cameraman_gaussian_filtered)
plt.title("Cameraman Smoothed with Gaussian Filter")
plt.show()

plt.imshow(cameraman_minus_gaussian_filtered, vmin=0, vmax=1)
plt.title("Cameraman Minus Gaussian Filtered Cameraman")
plt.show()

cameraman_sharpened = cameraman_minus_gaussian_filtered + cameraman
plt.imshow(cameraman_sharpened, vmin=0, vmax=1)
plt.title("Cameraman with Unsharp Masking Filter")
plt.show()

k = 0.5
cameraman_sharpened_2 = cameraman + k * cameraman_minus_gaussian_filtered
plt.imshow(cameraman_sharpened_2, vmin=0, vmax=1)
plt.title("Cameraman Edge Enhancing k = 0.5")
plt.show()

k = 50
cameraman_sharpened_3 = cameraman + k * cameraman_minus_gaussian_filtered
plt.imshow(cameraman_sharpened_3, vmin=0, vmax=1)
plt.title(f"Cameraman High Boost Filtering k = {k}")
plt.show()