import skimage as sk
import skimage.filters
import numpy as np

def hybrid_image(img1, img2, cutoff_low, cutoff_high, intensity_factor):
    """
    creates a hybrid image by combining the high frequencies of img2
    with the low frequencies of img1

    img1 and img2 need to be the same shape
    """
    low_pass = skimage.filters.gaussian(img1, sigma = cutoff_low)

    img2 = skimage.color.rgb2gray(img2)
    img2 = skimage.util.invert(img2)
    blurred2 = skimage.filters.gaussian(img2, sigma = cutoff_high)
    fft_blurred2 = np.fft.fft2(blurred2)
    fft_img2 = np.fft.fft2(img2)
    fft_high_pass = np.subtract(fft_blurred2, fft_img2)
    high_pass = np.fft.ifft2(fft_high_pass).real
    high_pass = np.multiply(high_pass, intensity_factor)
    high_pass = skimage.color.gray2rgb(high_pass)

    hybrid = np.add(low_pass, high_pass)
    return hybrid
