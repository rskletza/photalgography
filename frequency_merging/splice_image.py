import numpy as np
import skimage as sk
import skimage.filters
import skimage.util

def stacks(img, n):
    """
    returns an array with n frequency bands of the same size of an image (laplacian filter)
    """
    bands = []
    last_img = img
    for i in range(1,n+1):
        sigma = np.power(2,i)
        filtered = skimage.filters.gaussian(img, sigma = sigma)
        band = np.subtract(last_img, filtered)
        bands.append(band)
        last_img = filtered

    bands.append(last_img)
    return bands

def rebuild_stacks(bands):
    """
    returns an image constructed from frequency bands (laplacian filter)
    """
    result = np.zeros(bands[0].shape)
    for laplacian in bands:
        fft_laplacian = np.fft.fft2(laplacian)
        result = np.add(result, fft_laplacian)

    result = np.fft.ifft2(result).real
    result = sk.exposure.rescale_intensity(result, in_range=(-1.0,1.0))
    return result

def sample(img, n):
    """
    returns a reduced image (achieved by sampling every nth pixel) 
    """
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
    """
    returns a blown up image (lossy reconstruction of sampled image)
    """
    #insert a copy of each pixel next to the pixel
    img = np.repeat(img, n, axis=1)
    #insert a copy of each row underneath the row
    img = np.repeat(img, n, axis=0)
    return img

def pyramid(img, n=5):
    """
    returns an array with n frequency bands of an image (laplacian filter) with diminishing size
    """
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
    """
    returns an image constructed from frequency bands (laplacian filter) saved as a pyramid
    reconstruction is lossy, the parameter s determines the strength of the anti-aliasing
    """
    current = bands.pop()
    while len(bands) > 0:
        next_band = bands.pop()
        current = skimage.transform.resize(current, next_band.shape)
        fft_current = np.add(np.fft.fft2(current), np.fft.fft2(next_band))
        current = np.fft.ifft2(fft_current).real
        current = upsample(current, 2)
        current = skimage.filters.gaussian(current, sigma=s)

    result = sk.exposure.rescale_intensity(current, in_range=(-1.0,1.0))
    return result

def splice(img1, img2, mask, n):
    """
    splices two images together by splicing their respective frequency bands
    the edge is defined by the mask
    """
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

    result = rebuild_stacks(merged_bands)
    return result
