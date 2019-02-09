from align_images import align_images
from crop_image import crop_image
#from hybrid_image import hybrid_image
from stacks import stacks
import skimage as sk
import skimage.filters
import skimage.io as skio
import numpy as np

def hybrid_image(img1, img2, cutoff_low, cutoff_high):
    low_pass = skimage.filters.gaussian(img1, sigma = cutoff_low)

    img2 = skimage.color.rgb2gray(img2)
    img2 = skimage.util.invert(img2)
    blurred2 = skimage.filters.gaussian(img2, sigma = cutoff_high)
    fft_blurred2 = np.fft.fft2(blurred2)
    fft_img2 = np.fft.fft2(img2)
    fft_high_pass = np.subtract(fft_blurred2, fft_img2)
    high_pass = np.fft.ifft2(fft_high_pass).real
    high_pass = np.multiply(high_pass, 0.9)
    high_pass = skimage.color.gray2rgb(high_pass)
    hybrid = np.add(low_pass, high_pass)

    return hybrid

# read images
#im1 = skio.imread('./originals/Marilyn_Monroe.png')
#im2 = skio.imread('./originals/Albert_Einstein.png')
#im2 = skio.imread('./originals/rosalie_altes-bild.jpg')
#im1 = skio.imread('./originals/gr2-profile-pic.jpg')
#im2 = skio.imread('./originals/pigeon.jpg', mode='L')
#im1 = skio.imread('./originals/eagle.jpg', mode='L')
#im2 = skio.imread('./originals/tiger2_fade.jpg')
#im1 = skio.imread('./originals/kitten.jpg')
im2 = skio.imread('./originals/seena.jpg')
im1 = skio.imread('./originals/puschl.jpg')
im1 = sk.img_as_float(im1)
im2 = sk.img_as_float(im2)

# use this if you want to align the two images (e.g., by the eyes) and crop
# them to be of same size
im2, im1 = align_images(im2, im1)

# Choose the cutoff frequencies and compute the hybrid image (you supply
# this code)
arbitrary_value_1 = 2
arbitrary_value_2 = 100
cutoff_low = arbitrary_value_1
cutoff_high = arbitrary_value_2
im12 = hybrid_image(im1, im2, cutoff_low, cutoff_high)
print("show uncropped")
skio.imshow(im12)
skio.show()

# Crop resulting image (optional)
assert im12 is not None, "im12 is empty, implement hybrid_image!"
im12 = crop_image(im12)

im12 = sk.exposure.rescale_intensity(im12, in_range=(-1.0,1.0))

    #uncomment to save image
#name = os.path.basename(name)
#name = os.path.splitext(name)[0]
skio.imsave("merged/cats.jpg", im12)

## Compute and display Gaussian and Laplacian Stacks (you supply this code)
#n = 5  # number of pyramid levels (you may use more or fewer, as needed)
#stacks(im12, n)
