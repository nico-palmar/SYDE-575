from skimage.color import rgb2ycbcr, rgb2lab, rgb2gray, ycbcr2rgb
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from scipy.fftpack import dct
from skimage.metrics import peak_signal_noise_ratio as PSNR

import skimage.transform as tf

# part 5
Z = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

def sub2ind(n_row, row, col):
  return n_row * col + row

def dctmtx(N):
  return dct(np.eye(N), norm='ortho', axis=0)

def func(x, mat):
  return mat @ x @ mat.T

def func1(x, mat):
  return np.multiply(mat, x)

def blockproc(im, mat, block_sz, func):
    h, w = im.shape
    m, n = block_sz
    im_out = np.zeros_like(im)
    for x in range(0, h, m):
        for y in range(0, w, n):
            block = im[x:x+m, y:y+n]
            im_out[x:x+m, y:y+n] = func(block, mat)
    return im_out

def reduce_resolution(image, title, factor):
    """
    reduce each image resolution by a factor of 2 (horizontally and vertically)
    """
    new_shape = image.shape[0] // factor, image.shape[1] // factor
    print(f"Old shape: {image.shape}, New shape: {new_shape}")
    image = tf.resize(image, new_shape, order=1)
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.show()
    print(image.shape)
    return image

def digital_zoom(image, factor, order, title):
    """
    Perform digital zoom where order NN = 0, bilinear = 1, bicubic = 3
    """
    new_shape = image.shape[0] * factor, image.shape[1] * factor
    # print(f"Old shape: {image.shape}, New shape: {new_shape}")
    image = tf.resize(image, new_shape, order=order)
    plt.imshow(image, cmap = 'gray')
    plt.title(title)
    plt.show()
    return image

def plot_dct_reconstructed_image(img, compression_coeff, quantization_matrix = Z, subimg_dim = 8):
    # note: assumes image scaled [0, 255]
    dct_matrix = dctmtx(subimg_dim)
    img_dct = np.floor(blockproc(img-128, dct_matrix, [subimg_dim, subimg_dim], func))
    Z_scaled = compression_coeff * quantization_matrix

    tile_amount = int(img.shape[0] / subimg_dim)
    Z_scaled = np.tile(Z_scaled, (tile_amount, tile_amount))
    img_dct_quantized = np.round(img_dct / Z_scaled).astype(int)
    print(f"The number of non-zero coefficients after DCT quantization is: {np.count_nonzero(img_dct_quantized)} / {(img.shape[0] * img.shape[0])} total coefficients")

    # reconstruct the image
    img_dct_recovered = img_dct_quantized * Z_scaled
    img_recovered = blockproc(img_dct_recovered, dct_T_matrix.T, [subimg_dim, subimg_dim], func) + 128

    plt.imshow(img_recovered, cmap="gray")
    plt.title(f"Compressed Image Reconstruction With Compression Factor {compression_coeff}")
    plt.show()

    print(f"The PSNR for the reconstruction with compression factor = {compression_coeff} is {round(PSNR(img, img_recovered, data_range=255), 3)}")

original_peppers = imread('../lab_images/peppers.png')
peppers = rgb2ycbcr(imread('../lab_images/peppers.png'))

luma = peppers[..., 0]
cb = peppers[..., 1]
cr = peppers[..., 2]

# plot pepper luma and chroma channels
plt.figure(figsize=(14, 8))
plt.subplot(1, 3, 1)

plt.title("Y")
plt.imshow(luma, cmap = 'gray')

plt.subplot(1, 3, 2)

plt.title("Cb")
plt.imshow(cb, cmap = 'gray')

plt.subplot(1, 3, 3)

plt.title("Cr")
plt.imshow(cr, cmap = 'gray')

plt.show()

peppers_reduced_cb = reduce_resolution(cb, "Peppers Reduced Resolution Cb", 2)
peppers_reduced_cr = reduce_resolution(cr, "Peppers Reduced Resolution Cr", 2)
peppers_upsampled_cb = digital_zoom(peppers_reduced_cb, 2, 1, "Peppers Upsampled Resolution Cb")
peppers_upsampled_cr = digital_zoom(peppers_reduced_cr, 2, 1, "Peppers Upsampled Resolution Cr")

# recombine the Y with upsampled Cb and Cr to create new image
recombined_chroma_peppers = np.stack([luma, peppers_upsampled_cb, peppers_upsampled_cr])
recombined_chroma_peppers = np.transpose(recombined_chroma_peppers, axes=(1, 2, 0))
recombined_chroma_peppers_rgb = ycbcr2rgb(recombined_chroma_peppers)

plt.figure(figsize=(14, 8))
plt.subplot(1, 2, 1)
plt.imshow(original_peppers)
plt.title('Original Peppers Image')

plt.subplot(1, 2, 2)
plt.imshow(recombined_chroma_peppers_rgb)
plt.title('Recombined Peppers Upsampled Cb and Cr')
plt.show()

peppers_reduced_luma = reduce_resolution(luma, "Peppers Reduced Resolution Luma", 2)
peppers_upsampled_luma = digital_zoom(peppers_reduced_luma, 2, 1, "Peppers Upsampled Resolution Luma")

recombined_luma_peppers = np.stack([peppers_upsampled_luma, cb, cr])
recombined_luma_peppers = np.transpose(recombined_luma_peppers, axes=(1, 2, 0))
recombined_luma_peppers_rgb = ycbcr2rgb(recombined_luma_peppers)
recombined_luma_peppers.shape


plt.figure(figsize=(14, 8))
plt.subplot(1, 2, 1)
plt.imshow(original_peppers)
plt.title('Original Peppers Image')

plt.subplot(1, 2, 2)
plt.imshow(recombined_luma_peppers_rgb)
plt.title('Recombined Peppers Upsampled Luma')
plt.show()

p_lab = rgb2lab(original_peppers)

# Function to initialize cluster centers
def mu(lab_image, row, col):
    """Extract initial cluster centers based on row and column indices."""
    mu = lab_image[row,col]
    return mu

# Reshape the L*a*b* channels
m, n, ch = p_lab.shape
reshape_p_lab = np.reshape(p_lab, (m * n, ch), order='F')

# Perform KMeans clustering for K = 2
row_k2 = np.array([55, 200]) - 1
col_k2 = np.array([155, 400]) - 1

mu_k2 = mu(p_lab, row_k2, col_k2)

kmeans_k2 = KMeans(n_clusters=2, init=mu_k2, random_state=42)
cluster_idx_k2 = kmeans_k2.fit_predict(reshape_p_lab)

# Reshape the cluster indices to match the original image dimensions
segmented_k2 = np.reshape(cluster_idx_k2, (m, n), order='F')

plt.imshow(segmented_k2, cmap='jet')
plt.title('K-Means Segmentation (K=2)')
plt.show()

# Perform KMeans clustering for K = 4
row_k4 = np.array([55, 130, 200, 280]) - 1
col_k4 = np.array([155, 110, 400, 470]) - 1

mu_k4 = mu(p_lab, row_k4, col_k4)

kmeans_k4 = KMeans(n_clusters=4, init=mu_k4, random_state=42)
cluster_idx_k4 = kmeans_k4.fit_predict(reshape_p_lab)

segmented_k4 = np.reshape(cluster_idx_k4, (m, n), order='F')

plt.imshow(segmented_k4, cmap='jet')
plt.title('K-Means Segmentation (K=4)')
plt.show()

segmented_images_k4 = []

for cluster_label in range(4):  # Loop through each cluster
    # Create a mask for the current cluster
    mask = (segmented_k4 == cluster_label)
    
    # Initialize an empty image with the same shape as the original
    segmented_image = np.zeros_like(original_peppers)
    
    # Apply the mask to the original image
    for c in range(3):  # Loop through each color channel (R, G, B)
        segmented_image[..., c] = original_peppers[..., c] * mask
    
    segmented_images_k4.append(segmented_image)
    
    # Display the segmented region
    plt.imshow(segmented_image)
    plt.title(f'Segmented Region for Cluster {cluster_label + 1}')
    plt.axis('off')
    plt.show()
  
# create the DCT matrix
dct_T_matrix = dctmtx(8)
plt.imshow(dct_T_matrix)
plt.title("8x8 DCT Matrix")
plt.show()

# plot each row as a function
for i, row in enumerate(dct_T_matrix):
    plt.plot(np.arange(0, 8), row, label=f"Row {i}")
    plt.title("Rows of the DCT Matrix")
    plt.legend()
plt.show()
  
lena = (rgb2gray(imread('../lab_images/lena.tiff')) * 255).astype(int)
lena_dct = np.floor(blockproc(lena-128, dct_T_matrix, [8, 8], func))  
first_dct_patch = np.abs(lena_dct[80:80+8, 296: 296+8])
second_dct_patch = np.abs(lena_dct[0: 8, 0: 8])

plt.imshow(first_dct_patch)
plt.title("DCT Transformed Patch At (80, 296)")
plt.show()
plt.imshow(lena[80:80+8, 296: 296+8], cmap="gray")
plt.title("Lena Subimage At (80, 296)")
plt.show()

plt.imshow(second_dct_patch)
plt.title("DCT Transformed Patch At (0, 0)")
plt.show()
plt.imshow(lena[0: 8, 0: 8], cmap="gray")
plt.title("Lena Subimage At (0, 0)")
plt.show()

mask = np.zeros((8 , 8 ))
mask [ 0 , 0 ] = 1
mask [ 0 , 1 ] = 1
mask [ 0 , 2 ] = 1
mask [ 1 , 0 ] = 1
mask [ 1 , 1 ] = 1
mask [ 2 , 0 ] = 1

plt.imshow(mask, cmap="gray")
plt.title("Mask Keeping Only Low Freq Components")
plt.show()

lena_dct_thresh = blockproc(lena_dct , mask , [8 , 8] , func1)
lena_thresh = np.floor(blockproc(lena_dct_thresh, dct_T_matrix.T, [8, 8], func)) + 128

plt.figure(figsize=(6, 6))
plt.imshow(lena_thresh, cmap="gray")
plt.title("Lena Reconstructed With Only Low Freq Components of DCT")
plt.show()
print(f"The PSNR between original Lena and reconstructed Lena using only low freq DCT components is {round(PSNR(lena, lena_thresh, data_range=255), 3)}")


plot_dct_reconstructed_image(lena, 1)
plot_dct_reconstructed_image(lena, 3)
plot_dct_reconstructed_image(lena, 5)
plot_dct_reconstructed_image(lena, 10)
