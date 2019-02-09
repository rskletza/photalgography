import sys
import os
import numpy as np
import skimage as sk
import skimage.io as skio
import skimage.filters
import skimage.util
import matplotlib.pyplot as plt

def stacks(img, n):
    bands = []
    last_img = img
    for i in range(1,n+1):
        sigma = np.power(2,i)
        filtered = skimage.filters.gaussian(img, sigma = sigma)
        band = np.subtract(last_img, filtered)
        bands.append(band)
        last_img = filtered

    bands.append(last_img)
    plt.axis("off")
    fig = plt.figure(1,(len(bands)))
    k = 1
    for band in bands:
        ax = fig.add_subplot(1,len(bands), k)
        ax.imshow(np.add(band,0.5))
        ax.axis('off')
        k += 1
    plt.show()
    return bands

def rebuild_stacks(bands):
    result = np.zeros(bands[0].shape)
    for laplacian in bands:
        fft_laplacian = np.fft.fft2(laplacian)
        result = np.add(result, fft_laplacian)
#        skio.imshow(np.fft.ifft2(result).real)
#        skio.show()

    result = np.fft.ifft2(result).real
    result = sk.exposure.rescale_intensity(result, in_range=(-1.0,1.0))
    return result


def merge(img1, img2, mask, n):
    bands1 = stacks(img1, n)
    bands2 = stacks(img2, n)
    merged_bands = []
    for i in range(n):
        blurred_mask = skimage.filters.gaussian(mask, sigma = np.power(2,i))
        masked_band1 = np.multiply(bands1[i], blurred_mask)
#        skio.imshow(masked_band1)
#        skio.show()
        masked_band2 = np.multiply(bands2[i], skimage.util.invert(blurred_mask))
        merged = np.add(np.fft.fft2(masked_band1), np.fft.fft2(masked_band2))
        merged = np.fft.ifft2(merged).real
#        skio.imshow(np.add(merged, 0.5))
#        skio.show()
        merged_bands.append(merged)

    plt.axis("off")
    fig = plt.figure(1,(len(merged_bands)))
    k = 1
    for band in merged_bands:
        ax = fig.add_subplot(1,len(merged_bands), k)
        ax.imshow(np.add(band,0.5))
        ax.axis('off')
        k += 1
    plt.show()
    result = rebuild_stacks(merged_bands)
    return result

img1 = skio.imread("./originals/apple.jpeg")
img2 = skio.imread("./originals/orange.jpeg")
mask = skio.imread("./masks/apple-orange-mask.jpg")
img1 = sk.img_as_float(img1)
img2 = sk.img_as_float(img2)
mask = sk.img_as_float(mask)

merged = merge(img1, img2, mask, 10)
skio.imshow(np.add(merged,0.3))
skio.show()

#loop through all the files given as arguments
#for name in sys.argv[1:]:
#    img = skio.imread(name)
#    img = sk.img_as_float(img)
#
#    bands = stacks(img, 5)
#    plt.axis("off")
#    fig = plt.figure(1,(len(bands)))
#    k = 1
#    for band in bands:
#        ax = fig.add_subplot(1,len(bands), k)
#        ax.imshow(np.add(band,0.5))
#        ax.axis('off')
#        k += 1
#    plt.show()
#
#    image = rebuild_stacks(bands)
#    skio.imshow(image)
#    skio.show()


    #uncomment to save image
#    name = os.path.basename(name)
#    name = os.path.splitext(name)[0]
#    skio.imsave("out_" + name + ".jpg", img_out)
