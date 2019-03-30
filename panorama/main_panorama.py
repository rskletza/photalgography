import sys
import numpy as np
import skimage.io as skio
import skimage as sk
import time
import matplotlib.pyplot as plt

import panorama

if not sys.stdin.isatty():
    arguments = sys.stdin.read()
    arguments = arguments.split(" ")
    #remove newline character at end of input
    arguments[-1] = arguments[-1][:-1]

else:
    arguments = sys.argv[1:]

if len(arguments) < 4:
    print("please specify at least two images and their respective points in the correct order (image1 points12_1 points12_2 image2 points23_2 points23_3 image3 ... imageN points1 points2 points3 ... imageN). The center image will be used as the reference image")
    sys.exit()

else:
#    if len(arguments) % 2 != 0:
#        print("the number of images and points does not match! please make sure all images have a point file and vice versa")
#        sys.exit()

    arguments.insert(0, "dummy")
    arguments.append("dummy")
    img_number = len(arguments)/3
    img_sets = np.split(np.array(arguments), img_number)
    base_index = int(len(img_sets)/2)

    count = 0
    right_pointer = base_index + 1
    left_pointer = base_index - 1

    left_points = panorama.parse_pointfile(img_sets[base_index][0])
    base_image = skio.imread(img_sets[base_index][1])
    base_image = sk.img_as_float(base_image)
    right_points = panorama.parse_pointfile(img_sets[base_index][2])
    base_set = panorama.PanoSet(base_image, left_points, right_points)

    #TODO check border cases!
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

        other_l_points = panorama.parse_pointfile(img_sets[i][0])
        other_img = skio.imread(img_sets[i][1])
        other_img = sk.img_as_float(other_img)
        other_r_points = panorama.parse_pointfile(img_sets[i][2])
        other_set = panorama.PanoSet(other_img, other_l_points, other_r_points)

        base_set = panorama.createPanorama(base_set, other_set, flag)
        count += 1

        skio.imshow(base_set.img)
        skio.show()
        skio.imsave("pano" + str(count) + ".jpg", base_set.img)

sys.exit()


img_left = skio.imread("./images/0-Rechauffement/pouliot.jpg")
img_left = sk.img_as_float(img_left)
H1 = [[0.9752, 0.0013, -100.3164], [-0.4886, 1.7240, 24.8480], [-0.0016, 0.0004, 1.0000]]
H2 = [[0.1814, 0.7402, 34.3412], [1.0209, 0.1534, 60.3258], [0.0005, 0, 1.0000]]
out_set = panorama.applyHomographyToImg(H2, panorama.PanoSet(img_left, [], []))
#skio.imsave("H2.jpg", out_set.img)
skio.imshow(out_set.img)
skio.show()
