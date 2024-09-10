from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage import filters
from skimage.exposure import equalize_hist
import matplotlib.pyplot as plt
import numpy as np

A = imread('cameraman.tif')

plt.figure()
plt.imshow(A, cmap='gray')
plt.show()

plt.hist(A.flatten(), bins = 10000)
hist, bins  = np.histogram(A, bins=10000)
plt.plot(bins[1:], hist)
plt.show()


m = np.mean(A)
s = np.std(A)
print("mean = %.2f, std = %.2f"%(m, s))

def imadjust(x,a,b,c,d,gamma=1):
# with gamma < 1
# or gamma > 1
    return (((x-a)/(b-a))**gamma)*(d-c)+c
B = imadjust(A, A.min(), A.max(), 0, 1, 2)
plt.imshow(B, cmap='gray')
plt.show()

B = equalize_hist(A)
plt.imshow(B, cmap='gray')
plt.show()

kernel_size = 3 # Adjust this for different filter sizes
C = filters.rank.mean(A, np.ones((kernel_size, kernel_size)))
C = C / 255
plt.imshow(C, cmap='gray')
plt.show()

D = A.astype('float') / 255 - C
plt.imshow(D + 0.5, cmap='gray')
plt.show()

E = A.astype('float') / 255 + D
plt.imshow(E, cmap='gray', vmin=0, vmax=1)
plt.show()

F = filters.sobel_h(A) / 255

G = filters.sobel_v(A) / 255

I = np.sqrt(F**2 + G**2)
plt.imshow(I, cmap='gray')
plt.show()