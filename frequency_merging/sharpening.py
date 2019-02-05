import numpy as np
import skimage as sk
import skimage.io as skio
import skimage.filters
import sys
import skimage.color

def sharpen(img, intensity=1.5):
    #details = sk.color.rgb2grey(img)
    blurred = sk.filters.gaussian(img, sigma=2)
    details = np.subtract(img, blurred)
    #details = sk.color.grey2rgb(details)
    details = np.multiply(details, intensity)
    skio.imshow(details)
    skio.show()
    img = np.add(img, details)
    return img
    

#loop through all the files given as arguments
for name in sys.argv[1:]:
    img = skio.imread(name)
    
    img_orig = sk.img_as_float(img)

    img = sharpen(img_orig)

    #uncomment to view original image
    #skio.imshow(img_orig)
    #skio.show()

    skio.imshow(img)
    skio.show()
    
    #uncomment to save image
#    name = os.path.basename(name)
#    name = os.path.splitext(name)[0]
#    skio.imsave("out_" + name + ".jpg", img_out)
