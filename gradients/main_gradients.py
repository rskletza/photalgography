import sys
import os
import numpy as np
from skimage import io, color
import skimage as sk
import time
import matplotlib.pyplot as plt

import gradientshop

if __name__ == "__main__":
    arguments = sys.argv[1:]

#    fct = gradientshop.basic_sharpening
#    fct = gradientshop.salient_sharpening
    fct = gradientshop.npr_filter

    name = arguments[0]
    print(name)
    img = sk.img_as_float(io.imread(name))
    start = time.time()
    if len(arguments) > 1:
        params = np.array([np.load(n) for n in arguments[1:]])
        out = gradientshop.filter_img(img, fct, params)
    else:
        out = gradientshop.filter_img(img, fct)
    end = time.time()
    print(str(end - start) + "s")

    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(img)
    axarr[1].imshow(out)
    plt.show()

    filename, ext = os.path.splitext(os.path.split(name)[1])
    io.imsave("./npr_results/" + filename + "_npr_lines" + ext, np.clip(out, 0, 1))
