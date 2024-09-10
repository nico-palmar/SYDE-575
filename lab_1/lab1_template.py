from skimage.color import rgb2gray
from skimage.io import imread
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure
plt.gray()

lena= rgb2gray(imread('lena.tiff')) *255
cameraman = imread('cameraman.tif').astype(np.float64)
tire = imread('tire.tif').astype(np.float64) / 255.0
