import sys
import numpy as np
import skimage.io as skio
import skimage as sk
import merge
from hybrid_image import hybrid_image_separate

def calc_dissolve(x):
    """
    function to calculate a good dissolve_frac for a specific warp_frac
    """
    if x <= 0.5:
        return 2*np.power(x,2)
    else:
        return 2.0/3.0 * np.power(x,2) + 1.0/3.0

if len(sys.argv) != 5:
    print("please give the images you want to combine along with the corresponding points. The order is image1 points1 image2 points2")
    sys.exit()

im1 = skio.imread(sys.argv[1])
im2 = skio.imread(sys.argv[3])

points1 = merge.parse_pointfile(sys.argv[2])
points1 = merge.add_image_edge_points(points1, im1)
points2 = merge.parse_pointfile(sys.argv[4])
points2 = merge.add_image_edge_points(points2, im2)

im1 = sk.img_as_float(im1)
im2 = sk.img_as_float(im2)

triangles = merge.calculate_triangles(points1, points2)

##create a morphed hybrid image
#hybrid_low, hybrid_high = hybrid_image_separate(im1, im2, 4, 10, 0.8)
#morphed = merge.morph(hybrid_low, hybrid_high, points1, points2, triangles, 0.5, -1)
#skio.imshow(morphed)
#skio.show()
#morphed = sk.exposure.rescale_intensity(morphed, in_range=(-1.0,1.0))
##skio.imsave("./hybrid_morphed.jpg", morphed)

#create a 50/50 morph of two images (warp, diss)
morphed = merge.morph(im1, im2, points1, points2, triangles, 0.7, 0.3)
skio.imshow(morphed)
skio.show()
skio.imsave("./femininisation_0703.jpg", morphed)

#uncomment to create a morph sequence, n specifies the number of images in the sequence
#n = 60
#i = 0
#warps = np.linspace(0,1,n)
#
#diss = []
#for x in warps:
#    diss.append(calc_dissolve(x))
#
#for warp_frac, dissolve_frac in zip(warps, diss):
#    morphed = merge.morph(im1, im2, points1, points2, triangles, warp_frac, dissolve_frac)
#    skio.imsave("./" + str(i).zfill(3) + ".jpg", morphed)
#    i += 1
