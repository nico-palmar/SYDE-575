import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector
import skimage.util
import scipy.ndimage as ndimage
from skimage.color import rgb2gray
from skimage.morphology import disk
from skimage.io import imread
import matplotlib
import skimage.restoration as restoration

import skimage.filters.rank as rank

plt.gray()
def PSNR(f,g):
    return 10*np.log10(1.0/ np.mean(np.square(f-g)))


class Selector:
    def __init__(self, ax):
        self.RS = RectangleSelector(ax, self.line_select_callback,
                                     useblit=True,
                                       button=[1, 3],  
                                       minspanx=5, minspany=5,
                                       spancoords='pixels',
                                       interactive=True)
        self.bbox = [None, None, None, None]
        
    def line_select_callback(self,eclick, erelease):
        'eclick and erelease are the press and release events'
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        self.bbox = [int(y1), int(y2), int(x1), int(x2)]
    def get_bbox(self):
        return self.bbox

f = imread('../lab_images/cameraman.tif').astype(np.float64)/255
I = imread('../lab_images/degraded.tif').astype(np.float64)/255

# Display the f_blur image
plt.imshow(f, cmap='gray')
plt.title(f'Original Cameraman Image')
plt.show()

h_d = disk(4)
h = np.zeros((256,256))
h[0:9,0:9] = h_d
h = np.roll(h, (-5,-5)) / np.sum(h)
h_freq = np.fft.fft2(h)
f_blurfreq = h_freq*np.fft.fft2(f)
f_blur = np.real(np.fft.ifft2(f_blurfreq))

psnr_blurred = PSNR(f, f_blur)

# Display the f_blur image
plt.imshow(f_blur, cmap='gray')
plt.title(f'Blurred Cameraman Image (PSNR: {psnr_blurred:.2f} dB)')
plt.show()

f_restored = np.real(np.fft.ifft2(f_blurfreq / h_freq))
psnr_restored = PSNR(f, f_restored)

# Display the f_blur image
plt.imshow(f_restored, cmap='gray')
plt.title(f'Restored Cameraman Image (PSNR: {psnr_restored:.2f} dB)')
plt.show()

# Plot the original and denoised images
fig, ax = plt.subplots(1, 3, figsize=(16, 6))
ax[0].imshow(f, cmap='gray')
ax[0].set_title('Cameraman Image')
ax[1].imshow(f_blur, cmap='gray')
ax[1].set_title(f'Blurred Cameraman Image (PSNR: {psnr_blurred:.2f} dB)')
ax[2].imshow(f_restored, cmap='gray')
ax[2].set_title(f'Restored Cameraman Image (PSNR: {psnr_restored:.2f} dB)')
plt.show()

# 1. Compare the restored image with the original image and the blurred image. How does the restored
# image and the PSNR differ from the blurred image? Is it better or worse? Why?

f_blur_noise = skimage.util.random_noise(f_blur, var = 0.002)

f_freq_noisy = np.fft.fft2(f_blur_noise)
f_restored_freq = f_freq_noisy / h_freq
f_gaussian_restored = np.real(np.fft.ifft2(f_restored_freq))

psnr_gaussian_restored = PSNR(f, f_gaussian_restored)

# Display the restored image and PSNR
plt.imshow(f_gaussian_restored, cmap='gray')
plt.title(f'Restored Image from Gaussian Noise (PSNR: {psnr_gaussian_restored:.2f} dB)')
plt.show()

# 2. Compare the restored image with the restored image from the previous step. How does the restored
# image and the PSNR differ from the previous restored image? Is it better or worse? Why?
# 3. Can you draw any conclusions about inverse filtering when applied to noise degraded images?

wiener_psf = np.fft.fftshift(np.real(np.fft.ifft2(h_freq)))

f_wiener_restored = restoration.wiener(f_blur_noise, wiener_psf, 0.1)

psnr_wiener_restored = PSNR(f, f_wiener_restored)

# Display the restored image and PSNR
plt.imshow(f_wiener_restored, cmap='gray')
plt.title(f'Restored Image using Wiener (PSNR: {psnr_wiener_restored:.2f} dB)')
plt.show()

# 4. Compare the restored image with the restored image from the previous step. How does the restored
# image and the PSNR differ from the previous restored image? Is it better or worse? Why? Explain it
# in context with the concept behind Wiener filtering.
# 5. Can you draw any conclusions about Wiener filtering when applied to noise degraded images?

ax = plt.gca()
ax.imshow(I)
select = Selector(ax)
# Choose flat region on degraded image
plt.title('Degraded Image')
plt.show()

bbox = select.bbox
y1, y2, x1, x2 = bbox
print(bbox)
flat_region = I[y1:y2, x1:x2]
sigma_noise_sq = np.var(flat_region)

print(f"Estimated noise variance: {sigma_noise_sq:.6f}")

# Compute local mean and variance
mn = np.ones((5, 5)) / 25
local_mean = signal.convolve(I, mn, mode='same')
local_var = signal.convolve(I**2, mn, mode='same') - local_mean**2

# Compute matrix K for each pixel
K = (local_var - sigma_noise_sq) / (local_var + 1e-10)
K_max = np.maximum((local_var - sigma_noise_sq) / (local_var + 1e-10), 0)

# Apply Lee filter
lee_denoised_image = np.real(K * I + (1 - K) * local_mean)

lee_denoised_image_max = np.real(K_max * I + (1 - K_max) * local_mean)

psnr_lee_value = PSNR(f, lee_denoised_image)
psnr_lee_value_max = PSNR(f, lee_denoised_image_max)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(lee_denoised_image, cmap='gray')
ax[0].set_title(f'Denoised Image with Lee Filter (PSNR: {psnr_lee_value:.2f} dB)')
ax[1].imshow(lee_denoised_image_max, cmap='gray')
ax[1].set_title(f'Max Denoised Image with Lee Filter (PSNR: {psnr_lee_value_max:.2f} dB)')
plt.show()

# 6. Use a flat region of degraded.tif and estimate the variance of the noise (Use the provided Selector class, see the example usage in selector ex.py/selector ex.ipynb).

def gaussian_filter(n_rows, n_cols, stdv):
    """
    Returns a 2d Gaussian image filter.
    """
    g_r = signal.windows.gaussian(n_rows, stdv)
    g_c = signal.windows.gaussian(n_cols, stdv)

    G = np.outer(g_r, g_c)

    return G/np.max(G)

gaussian_low_pass = gaussian_filter(I.shape[0], I.shape[1], 30)

I_freq = np.fft.fftshift(np.fft.fft2(I))
I_gaussian_freq = I_freq * gaussian_low_pass
I_gaussian_low_pass = np.abs(np.fft.ifft2(I_gaussian_freq))

psnr_gaussian_low_pass = PSNR(f, I_gaussian_low_pass)

# Display the original and filtered images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(I, cmap='gray')
plt.title('Degraded Image')

plt.subplot(1, 2, 2)
plt.imshow(I_gaussian_low_pass, cmap='gray')
plt.title(f'Cameraman Image with Gaussian Low Pass Filter (PSNR: {psnr_gaussian_low_pass:.2f} dB)')
plt.show()

# 7. Compare this result to that using a Gaussian low pass filter with standard deviation of 30 (in the
# frequency domain). Note the performance in areas of high and low detail.

def lee_filter(img, filter_size, noisy_var):
    mn = np.ones((filter_size, filter_size))/ (filter_size*filter_size)
    local_mean = signal.convolve(img, mn, mode='same')
    local_var = signal.convolve(img**2, mn, mode='same') - local_mean**2
    K = np.maximum((local_var - noisy_var) / local_var, 0)
    denoised_lee_image = K * img + (1 - K) * local_mean
    plt.imshow(denoised_lee_image)  
    plt.title(f"Denoised Image with Lee Filtering, Filter Size = {filter_size}, Noisy Variance = {round(noisy_var, 3)}")
    plt.show()
    print(f"The PSNR between the original and lee filtered restored image with Filter Size = {filter_size}, Noisy Variance = {round(noisy_var, 3)} is {round(PSNR(f, denoised_lee_image), 2)} dB")
    return denoised_lee_image

lee_filter(I, 5, sigma_noise_sq)

lee_filter(I, 5, sigma_noise_sq/10)

lee_filter(I, 5, sigma_noise_sq*10)

# 8. Try varying your estimate of the noise variance both above and below the value you got from your flat
# region. How does this change the filter’s results? Why?

lee_filter(I, 5, sigma_noise_sq)

lee_filter(I, 3, sigma_noise_sq)

lee_filter(I, 7, sigma_noise_sq)


# 9. Try changing the size of the filter neighborhood to be smaller and larger than 5 × 5. How does this
# change the results? Why?