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

    result = np.fft.ifft2(result).real
    result = sk.exposure.rescale_intensity(result, in_range=(-1.0,1.0))
    return result

def sample(img, n):
    x_indices_set = set(range(img.shape[0]))
    x_indices_to_keep = set(range(1, img.shape[0],n))
    x_indices_to_delete = list(x_indices_set - x_indices_to_keep)
    img = np.delete(img,x_indices_to_delete, axis=0)

    y_indices_set = set(range(img.shape[1]))
    y_indices_to_keep = set(range(1, img.shape[1],n))
    y_indices_to_delete = list(y_indices_set - y_indices_to_keep)
    img = np.delete(img, y_indices_to_delete, axis=1)
    return img

def upsample(img, n):
    #insert a copy of each pixel next to the pixel
    img = np.repeat(img, n, axis=1)
    #insert a copy of each row underneath the row
    img = np.repeat(img, n, axis=0)
    return img

def pyramid(img, n=5):
    bands = []
    last_img = img
    for i in range(1,n+1):
        sigma = np.power(2,i)
        last_img = sample(last_img, 2)
        filtered = skimage.filters.gaussian(img, sigma = sigma)
        filtered = sample(filtered, sigma)
        if (filtered.shape != last_img.shape): #make sure there are no small offsets
            last_img = skimage.transform.resize(last_img, filtered.shape)
        band = np.subtract(last_img, filtered)
        bands.append(band)
        last_img = filtered

    bands.append(last_img)
    return bands

def rebuild_pyramid(bands, s=1):
    current = bands.pop()
    while len(bands) > 0:
        next_band = bands.pop()
        current = skimage.transform.resize(current, next_band.shape)

        fft_current = np.add(np.fft.fft2(current), np.fft.fft2(next_band))
        current = np.fft.ifft2(fft_current).real
        current = upsample(current, 2)
        current = skimage.filters.gaussian(current, sigma=s)
        skio.imshow(current)
        skio.show()
    result = sk.exposure.rescale_intensity(current, in_range=(-1.0,1.0))
    return result

def merge(img1, img2, mask, n):
    bands1 = stacks(img1, n)
    bands2 = stacks(img2, n)
    merged_bands = []
    for i in range(n+1):
        blurred_mask = skimage.filters.gaussian(mask, sigma = np.power(2,i))
        masked_band1 = np.multiply(bands1[i], blurred_mask)
        masked_band2 = np.multiply(bands2[i], skimage.util.invert(blurred_mask))
        merged = np.add(np.fft.fft2(masked_band1), np.fft.fft2(masked_band2))
        merged = np.fft.ifft2(merged).real
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

#img1 = skio.imread("./originals/apple.jpeg")
#img2 = skio.imread("./originals/orange.jpeg")
#mask = skio.imread("./masks/apple-orange-mask.jpg")
#img1 = skio.imread("./originals/puma_adjusted.jpg")
#img2 = skio.imread("./originals/kletzander.jpg")
#mask = skio.imread("./masks/pumaeyes.jpg")
#img1 = skio.imread("./originals/dani.JPG")
#img2 = skio.imread("./originals/ich_dani.JPG")
#mask = skio.imread("./masks/dani_ich.JPG")
#img1 = skio.imread("./originals/lion_adjusted.jpg")
#img2 = skio.imread("./originals/dandelion.jpg")
#mask = skio.imread("./masks/dandelion.jpg")
#img1 = skio.imread("./originals/faces_hand.jpg")
#img2 = skio.imread("./originals/hand.jpg")
#mask = skio.imread("./masks/hand2.jpg")
img1 = skio.imread("./originals/earth.jpg")
img2 = skio.imread("./originals/moon_big.jpg")
mask = skio.imread("./masks/mearth5.jpg")
#img1 = skio.imread("./originals/penguins_adjusted2.jpg")
#img2 = skio.imread("./originals/beach_adjusted.jpg")
#mask = skio.imread("./masks/beach2.jpg")
img1 = sk.img_as_float(img1)
img2 = sk.img_as_float(img2)
mask = sk.img_as_float(mask)

#img1 = skimage.color.rgb2gray(img1)
#img2 = skimage.color.rgb2gray(img2)
#mask = skimage.color.rgb2gray(mask)

#merged = merge(img1, img2, mask, 6)
#skio.imshow(merged)
#skio.show()

#loop through all the files given as arguments
for name in sys.argv[1:]:
    img = skio.imread(name)
    img = sk.img_as_float(img)

    bands = pyramid(img, 6)
    plt.axis("off")
    fig = plt.figure(1,(len(bands)))
    k = 1
    for band in bands:
        print(band.shape)
        ax = fig.add_subplot(1,len(bands), k)
        ax.imshow(np.add(band,0.5))
        ax.axis('off')
        k += 1
    plt.show()

    image = rebuild_pyramid(bands)
    skio.imshow(image)
    skio.show()
    skio.imsave("./tiggen.jpg", image)


    #uncomment to save image
#name = os.path.basename(mask)
#name = os.path.splitext(name)[0]
#skio.imsave("./spliced/mearth.jpg", merged)
