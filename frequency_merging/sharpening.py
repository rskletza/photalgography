import numpy as np
import skimage as sk
import skimage.io as skio
import skimage.filters
import skimage.exposure
import sys
import os
import skimage.color

def sharpen(img, intensity=1.5):
    blurred = sk.filters.gaussian(img, sigma=5)
    fft_img = np.fft.fft2(img) 
    fft_blurred = np.fft.fft2(blurred)
    fft_details = np.subtract(fft_img, fft_blurred)
    details = np.fft.ifft2(fft_details).real
    details = np.multiply(details, intensity)
    img = np.add(img, details)
    return img
    

#loop through all the files given as arguments
for name in sys.argv[1:]:
    img = skio.imread(name)
    
    img_orig = sk.img_as_float(img)

    img = sharpen(img_orig)
    img = sk.exposure.rescale_intensity(img, in_range=(-1.0,1.0))

    #uncomment to view original image
    #skio.imshow(img_orig)
    #skio.show()

    skio.imshow(img)
    skio.show()
    
    #uncomment to save image
    name = os.path.basename(name)
    name = os.path.splitext(name)[0]
    skio.imsave("sharpened/out_" + name + ".jpg", img)
