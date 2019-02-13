from align_images import align_images
from crop_image import crop_image
from hybrid_image import hybrid_image
import skimage as sk
import skimage.io as skio

# read images
#im1 = skio.imread('./originals/Marilyn_Monroe.png')
#im2 = skio.imread('./originals/Albert_Einstein.png')
#im2 = skio.imread('./originals/rosalie_altes-bild.jpg')
#im1 = skio.imread('./originals/gr2-profile-pic.jpg')
#im2 = skio.imread('./originals/pigeon.jpg', mode='L')
#im1 = skio.imread('./originals/eagle.jpg', mode='L')
#im2 = skio.imread('./originals/tiger2_fade.jpg')
#im1 = skio.imread('./originals/kitten.jpg')
#im2 = skio.imread('./originals/bobbycar.jpg')
#im1 = skio.imread('./originals/ferrari_gray.jpg')
#im2 = skio.imread('./originals/old_man.jpeg')
#im1 = skio.imread('./originals/matterhorn.jpg')
#im2 = skio.imread('./originals/wolf.jpg')
#im1 = skio.imread('./originals/1534638850129_gray.jpg')
#im2 = skio.imread('./originals/frown.JPG')
#im1 = skio.imread('./originals/smile_gray.JPG')
im2 = skio.imread('./originals/flo_finale.jpg')
im1 = skio.imread('./originals/me_boston.jpg')
im1 = sk.img_as_float(im1)
im2 = sk.img_as_float(im2)

# use this if you want to align the two images (e.g., by the eyes) and crop
# them to be of same size
im2, im1 = align_images(im2, im1)

# Choose the cutoff frequencies and compute the hybrid image (you supply
# this code)
arbitrary_value_1 = 7
arbitrary_value_2 = 10
cutoff_low = arbitrary_value_1
cutoff_high = arbitrary_value_2
im12 = hybrid_image(im1, im2, cutoff_low, cutoff_high)

# Crop resulting image (optional)
assert im12 is not None, "im12 is empty, implement hybrid_image!"
im12 = crop_image(im12)

im12 = sk.exposure.rescale_intensity(im12, in_range=(-1.0,1.0))

    #uncomment to save image
#name = os.path.basename(name)
#name = os.path.splitext(name)[0]
skio.imsave("merged/flo_me.jpg", im12)

## Compute and display Gaussian and Laplacian Stacks (you supply this code)
#n = 5  # number of pyramid levels (you may use more or fewer, as needed)
#stacks(im12, n)
