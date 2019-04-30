import sys
import numpy as np
from skimage import io, color
import skimage as sk
import time
import matplotlib.pyplot as plt

import gradientshop

if __name__ == "__main__":
    arguments = sys.argv[1:]

    for name in arguments:
        img = sk.img_as_float(io.imread(name))
        img = color.rgb2gray(img)
        print(img.shape)
#        gx, gy = gradientshop.get_gradients(img)
#        io.imshow(gx + gy)
#        io.show()
#        x_saliency, y_saliency = gradientshop.calc_saliency(img)
#        salient_gradients = np.abs(x_saliency + y_saliency)
#        salient_gradients = np.divide(salient_gradients, np.max(salient_gradients))
#        io.imshow(salient_gradients * 3)
#        io.show()
#        io.imsave("out.jpg", salient_gradients)
        out = gradientshop.filter_img(img, gradientshop.basic_sharpening)
        io.imsave("out.jpg", np.clip(out, 0, 1))
#        gradientshop.saliency_sharpening_noopt(img)

