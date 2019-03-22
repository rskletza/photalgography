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
    base_set = (left_points, base_image, right_points)

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
        other_set = (other_l_points, other_img, other_r_points)
        #print("other_set points (l, r): " + str(other_l_points) + str(other_r_points))

        base_set = panorama.createPanorama(base_set, other_set, flag)
        #print("new base image: (left_points, shape, right_points): " + str(base_set[0]) + str(base_set[1].shape) + str(base_set[2]))
        count += 1

#        skio.imsave("./intermed" + str(count) + ".jpg", base_set[1])
        skio.imshow(base_set[1])
        skio.show()

sys.exit()


img_left = skio.imread("./images/1-PartieManuelle/Serie1/IMG_2415.JPG")
img_left = sk.img_as_float(img_left)
pts_left_lc = panorama.parse_pointfile("./images/1-PartieManuelle/Serie1/pts_serie1/pts1_12.txt")
pts_center_lc = panorama.parse_pointfile("./images/1-PartieManuelle/Serie1/pts_serie1/pts2_12.txt")
img_center = skio.imread("./images/1-PartieManuelle/Serie1/IMG_2416.JPG")
img_center = sk.img_as_float(img_center)
pts_center_cr = panorama.parse_pointfile("./images/1-PartieManuelle/Serie1/pts_serie1/pts2_32.txt")
img_right = skio.imread("./images/1-PartieManuelle/Serie1/IMG_2417.JPG")
img_right = sk.img_as_float(img_right)
pts_right_cr = panorama.parse_pointfile("./images/1-PartieManuelle/Serie1/pts_serie1/pts3_32.txt")

left_set = ([], img_left, pts_left_lc)
center_set = (pts_center_lc, img_center, pts_center_cr)
right_set = (pts_right_cr, img_right, [])

plt.plot(center_set[2][:,0], center_set[2][:,1], 'o')
plt.plot(center_set[0][:,0], center_set[0][:,1], 'o')
plt.imshow(center_set[1])
plt.axis('equal')
plt.show()

plt.plot(left_set[2][:,0], left_set[2][:,1], 'o')
plt.imshow(left_set[1])
plt.axis('equal')
plt.show()

H = panorama.calcHomography(img_center, img_left, pts_center_lc, pts_left_lc)
intermed_set = panorama.applyHomography(H, left_set)

plt.plot(intermed_set[2][:,0], intermed_set[2][:,1], 'o')
plt.imshow(intermed_set[1])
plt.axis('equal')
plt.show()

out_set = panorama.align(center_set, intermed_set, 0)

plt.plot(out_set[0][:,0], out_set[0][:,1], 'o')
plt.plot(out_set[2][:,0], out_set[2][:,1], 'o')
plt.imshow(out_set[1])
plt.axis('equal')
plt.show()
#skio.imshow(out_img)
#skio.show()
#print(out_pts)
#skio.imsave("./intermed.jpg", out_img)
#out_img = skio.imread("./intermed.jpg")
#out_img = sk.img_as_float(out_img)
#out_pts = [[1070,  497], [1172,  505], [1258,  550], [ 902,  634], [1145,  740], [1221,  775], [1227,  633], [ 946,  769]] 
#combined, new_pts = panorama.align(img_center, out_img, pts_center_lc[0], out_pts)
#skio.imshow(combined)
#skio.show()

#H = panorama.calcHomography(img_center, img_right, pts_center_cr, pts_right_cr)
#out_img, out_pts = panorama.applyHomography(img_right, H, pts_right_cr)
#print(out_pts)
#skio.imsave("./intermed_right.jpg", out_img)
#out_pts = [[343, 507], [270, 546], [273, 657], [366, 696], [427, 748], [514, 483], [373, 479], [639, 707], [ 64, 418], [151, 462]]
#out_img = skio.imread("./intermed_right.jpg")
#out_img = sk.img_as_float(out_img)
#
#panorama.align(img_center, out_img, pts_center_cr[0], out_pts[0])


#H1 = [[0.9752, 0.0013, -100.3164], [-0.4886, 1.7240, 24.8480], [-0.0016, 0.0004, 1.0000]]
#H2 = [[0.1814, 0.7402, 34.3412], [1.0209, 0.1534, 60.3258], [0.0005, 0, 1.0000]]
#
#plt.plot(pts_left_lc[:,0], pts_left_lc[:,1], 'o')
#plt.imshow(img_left)
#plt.axis('equal')
#plt.show()
#
##start = time.time()
##out_img, out_pt = panorama.applyHomography(img_left, H2, pts_left_lc)
##print(out_pt)
##end = time.time()
##print(end - start)
#
#plt.plot(out_pt[:,0], out_pt[:,1], 'o')
#plt.imshow(out_img)
#plt.axis('equal')
#plt.show()



