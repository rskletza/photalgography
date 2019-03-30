import sys
import numpy as np
from skimage import io, color
import skimage as sk
import time
import matplotlib.pyplot as plt

import point_detection
import panorama

if __name__ == "__main__":
    if not sys.stdin.isatty():
        arguments = sys.stdin.read()
        arguments = arguments.split(" ")
        #remove newline character at end of input
        arguments[-1] = arguments[-1][:-1]

    else:
        arguments = sys.argv[1:]

    params = np.array([])
    dangling = np.array([])
    for i in range(len(arguments)-1):
        img1 = sk.img_as_float(io.imread(arguments[i]))
        if(img1.ndim == 2): #if image is grayscale
            img1_g = np.copy(img1)
            img1 = color.gray2rgb(img1)
        else:
            img1_g = color.rgb2gray(img1)

        img2 = sk.img_as_float(io.imread(arguments[i+1]))
        if(img2.ndim == 2): #if image is grayscale
            img2_g = np.copy(img2)
            img2 = color.gray2rgb(img2)
        else:
            img2_g = color.rgb2gray(img2)

        corrs1, corrs2 = point_detection.find_correspondences(img1_g, img2_g)
        if(i == 0):
            params = np.append(params, [dangling, img1, corrs1, corrs2])
        else:
            params = np.append(params, [img1, corrs1, corrs2])

    last_img = sk.img_as_float(io.imread(arguments[-1]))
    if(last_img.ndim == 2): #if image is grayscale
        last_img = color.gray2rgb(last_img)
    params = np.append(params, [last_img, dangling])
    np.save("params", params)
#    sys.exit()
#    params = np.load("params/params_japan.npy")

    img_number = len(params)/3
    img_sets = np.split(np.array(params), img_number)
    base_index = int(len(img_sets)/2)

    count = 0
    right_pointer = base_index + 1
    left_pointer = base_index - 1

    left_points = img_sets[base_index][0]
    base_image = img_sets[base_index][1]
    right_points = img_sets[base_index][2]
    base_set = panorama.PanoSet(base_image, left_points, right_points)

    while left_pointer >= 0 or right_pointer < len(img_sets):
        print(count)
        print("left_pointer: " + str(left_pointer))
        print("right_pointer: " + str(right_pointer))
        flag = count%2
        if flag == 0: #go left
            i = left_pointer
            left_pointer -= 1
        else:
            i = right_pointer
            right_pointer += 1

        other_l_points = img_sets[i][0]
        other_img = img_sets[i][1]
        other_r_points = img_sets[i][2]
        other_set = panorama.PanoSet(other_img, other_l_points, other_r_points)

        base_set = panorama.createPanorama(base_set, other_set, flag)
        count += 1

        io.imshow(base_set.img)
        io.show()
        io.imsave("pano" + str(count) + ".jpg", base_set.img)
    
