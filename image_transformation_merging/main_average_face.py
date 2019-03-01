import sys
import skimage as sk
import numpy as np
import merge
import skimage.io as skio
import matplotlib.pyplot as plt


if len(sys.argv) < 5:
    print("please specify at least two images and their respective points in the correct order (image1 image2 image3 ... imageN points1 points2 points3 ... imageN)")
    print(sys.argv[0])
    sys.exit()

else:
    arguments = sys.argv[1:]
    if len(arguments) % 2 != 0:
        print("the number of images and points does not match! please make sure all images have a point file and vice versa")
        sys.exit()

    img_pt_array = np.split(np.array(arguments), 2)
    tuple_array = list(zip(img_pt_array[0], img_pt_array[1]))
    pointnames = img_pt_array[1].tolist()
    print(pointnames)
    imgnames = img_pt_array[0].tolist()
    print(imgnames)

    im1 = skio.imread(imgnames[0])
    im1 = sk.img_as_float(im1)
#    im2 = skio.imread(sys.argv[3])

    #calculate average shape for all input images
    interpolated = merge.parse_pointfile(pointnames[0])
    points = [interpolated] #save parsed points for later use
    #while len(pointnames) != 0:
    for i in range(1, len(pointnames)):
        second = merge.parse_pointfile(pointnames[i])
        points.append(second) 
        interpolated = merge.interpolate_pointclouds(interpolated, second, 0.5)

    #add image edges
    interpolated = merge.add_image_edge_points(interpolated, im1)
    #calculate triangulation
    triangles = merge.calculate_triangles(interpolated)

#    plt.triplot(interpolated[:,0], interpolated[:,1], triangles)
#    plt.plot(interpolated[:,0], interpolated[:,1], 'o')
#    plt.imshow(im1)
#    plt.axis('equal')
#    plt.show()

    reshaped_faces = []
    print(list(zip(imgnames, pointnames)))
    #morph all faces into average shape and calculate average pixels
    for imgname, pts in zip(imgnames, points):
        img = skio.imread(imgname)
        img = sk.img_as_float(img)
        pts = merge.add_image_edge_points(pts, img)
        reshaped = merge.morph(img, img, pts, interpolated, triangles, 1, 0) #second parameter doesn't matter, as we are only taking pixels from the original image
#        skio.imshow(reshaped)
#        skio.show()
        reshaped_faces.append(reshaped)

    average_face = np.zeros(reshaped_faces[0].shape)
    for reshaped in reshaped_faces:
        average_face = np.add(average_face, reshaped)

    average_face = np.divide(average_face, len(reshaped_faces))
    skio.imshow(average_face)
    skio.show()

    skio.imsave("./average_utrecht.jpg", average_face)
