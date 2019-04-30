import sys
import numpy as np
from skimage import io, color
import skimage as sk
import time
import matplotlib.pyplot as plt

import gradientshop

if __name__ == "__main__":
    arguments = sys.argv[1:]

    i = 0
#    for name in arguments:
#        img = sk.img_as_float(io.imread(name))
##        img = color.rgb2gray(img)
##        img = np.linspace(0,1,16).reshape((4,4))
#        print(img.shape)
#        out = gradientshop.filter_img(img, gradientshop.salient_sharpening)
#        f, axarr = plt.subplots(1,2)
#        axarr[0].imshow(img)
#        axarr[1].imshow(out)
#        plt.show()
#        io.imsave("./salient_sharpening/" + str(i) + ".jpg", np.clip(out, 0, 1))
#        i += 1

    name = arguments[0]
    img = sk.img_as_float(io.imread(name))
    if len(arguments) > 1:
        params = np.array([np.load(n) for n in arguments[1:]])
        out = gradientshop.filter_img(img, gradientshop.salient_sharpening, params)
    else:
        out = gradientshop.filter_img(img, gradientshop.salient_sharpening)

    io.imsave("./salient_sharpening/" + str(i) + ".jpg", np.clip(out, 0, 1))
