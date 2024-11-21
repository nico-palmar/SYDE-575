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
    

def gaussian_filter(n_rows, n_cols, stdv):
    """
    Returns a 2d Gaussian image filter.
    """
    g_r = signal.windows.gaussian(n_rows, stdv)
    g_c = signal.windows.gaussian(n_cols, stdv)

    G = np.outer(g_r, g_c)

    return G/np.max(G)

# Create function for lee filter
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

f = imread('../lab_images/cameraman.tif').astype(np.float64)/255
I = imread('../lab_images/degraded.tif').astype(np.float64)/255

# Part 2 of the lab
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

# Display the blurred cameraman image
plt.imshow(f_blur, cmap='gray')
plt.title(f'Blurred Cameraman Image (PSNR: {psnr_blurred:.2f} dB)')
plt.show()

# Apply inverse filtering
f_restored = np.real(np.fft.ifft2(f_blurfreq / h_freq))
psnr_restored = PSNR(f, f_restored)

# Display the restored cameraman image
plt.imshow(f_restored, cmap='gray')
plt.title(f'Restored Cameraman Image (PSNR: {psnr_restored:.2f} dB)')
plt.show()

# Add zero mean gaussian noise to the blurred image
f_blur_noise = skimage.util.random_noise(f_blur, var = 0.002)
plt.title("Noisy Blurred Cameraman")
plt.imshow(f_blur_noise)
plt.show()

# Apply inverse filtering for the blur operation
f_freq_noisy = np.fft.fft2(f_blur_noise)
f_restored_freq = f_freq_noisy / h_freq
f_gaussian_restored = np.real(np.fft.ifft2(f_restored_freq))

psnr_gaussian_restored = PSNR(f, f_gaussian_restored)

# Display the restored image from Gaussian noise
plt.imshow(f_gaussian_restored, cmap='gray')
plt.title(f'Restored Image from Gaussian Noise (PSNR: {psnr_gaussian_restored:.2f} dB)')
plt.show()

# need to approx the noise to signal ratio
# note that the variance is 0.002, and image max is 1
avg = f_gaussian_restored.mean()
avg

# note that mean signal value is 0.465. The noise may be about 1 standard deviation away => sqrt(0.002)
standard_dev = np.sqrt(0.002)
standard_dev

nsr = standard_dev / avg
print(f"The noise to signal ratio is about {round(nsr, 2)}")

wiener_psf = np.fft.fftshift(np.real(np.fft.ifft2(h_freq)))

# Apply the Wiener filter on the noisy, blurred image
f_wiener_restored = restoration.wiener(f_blur_noise, wiener_psf, 0.1)

psnr_wiener_restored = PSNR(f, f_wiener_restored)

# Display the restored image and PSNR
plt.imshow(f_wiener_restored, cmap='gray')
plt.title(f'Restored Image using Wiener (PSNR: {psnr_wiener_restored:.2f} dB)')
plt.show()

# Part 3 of the lab
ax = plt.gca()
ax.imshow(I)
select = Selector(ax)
# Choose flat region on degraded image
plt.title('Degraded Cameraman Image')
plt.show()

bbox = select.bbox
y1, y2, x1, x2 = bbox
noisy_background_patch = I[y1:y2, x1:x2]
plt.imshow(noisy_background_patch)
plt.title("Background Noisy Patch")
plt.show()

sigma_noise_sq = np.var(noisy_background_patch)

print(f"Estimated noise variance: {sigma_noise_sq:.6f}")

lee_filter(I, 5, sigma_noise_sq)

# q7: compare the result to a gaussian LPF with sigma = 30
gaussian_low_pass = gaussian_filter(I.shape[0], I.shape[1], 30)

I_freq = np.fft.fftshift(np.fft.fft2(I))
I_gaussian_freq = I_freq * gaussian_low_pass

# Perform inverse Fourier transform to get the filtered image
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

# q8: change the variance both above and below what we computed from before and observe results
# note that the variance from before was about 0.01

lee_filter(I, 5, sigma_noise_sq)

lee_filter(I, 5, sigma_noise_sq/10)

lee_filter(I, 5, sigma_noise_sq*10)

# q9: change the filter size both above and below 5x5 and observe results

lee_filter(I, 5, sigma_noise_sq)

lee_filter(I, 3, sigma_noise_sq)

lee_filter(I, 7, sigma_noise_sq)

lee_filter(I, 15, sigma_noise_sq)
