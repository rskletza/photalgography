import sys
import skimage as sk
import numpy as np
import merge
import skimage.io as skio
import matplotlib.pyplot as plt

if not sys.stdin.isatty():
    arguments = sys.stdin.read()
    arguments = arguments.split(" ")
    #remove newline character at end of input
    arguments[-1] = arguments[-1][:-1]

else:
    arguments = sys.argv[1:]

if len(arguments) < 5:
    print("please specify at least two images and their respective points in the correct order (image1 image2 image3 ... imageN points1 points2 points3 ... imageN)")
    print(sys.argv[0])
    sys.exit()

else:
    if len(arguments) % 2 != 0:
        print("the number of images and points does not match! please make sure all images have a point file and vice versa")
        sys.exit()

    img_pt_array = np.split(np.array(arguments), 2)
    tuple_array = list(zip(img_pt_array[0], img_pt_array[1]))
    pointnames = img_pt_array[1].tolist()
    imgnames = img_pt_array[0].tolist()

    im1 = skio.imread(imgnames[0])
    im1 = sk.img_as_float(im1)

    #calculate average shape for all input images
    interpolated = merge.parse_pointfile(pointnames[0])
    pointsum = np.zeros(interpolated.shape)
    points = [] #save parsed points for later use
    for i in range(len(pointnames)):
        pointcloud = merge.parse_pointfile(pointnames[i])
        points.append(pointcloud) 

    interpolated = merge.average_pointclouds(points)
    #merge.write_pointfile(interpolated, "pointfile.txt")

    #add image edges
    interpolated = merge.add_image_edge_points(interpolated, im1)
    #calculate triangulation
    triangles = merge.calculate_triangles(interpolated)

    white = np.ones(im1.shape)

    plt.triplot(interpolated[:,0], interpolated[:,1], triangles)
    plt.plot(interpolated[:,0], interpolated[:,1], 'o')
    plt.imshow(white)
    plt.axis('equal')
    plt.show()

    reshaped_faces = []
    #morph all faces into average shape and calculate average pixels
    for imgname, pts in zip(imgnames, points):
        print(imgname)
        img = skio.imread(imgname)
        img = sk.img_as_float(img)
        pts = merge.add_image_edge_points(pts, img)
#        plt.triplot(pts[:,0], pts[:,1], triangles)
#        plt.plot(pts[:,0], pts[:,1], 'o')
#        plt.imshow(img)
#        plt.axis('equal')
#        plt.show()
        try:
            reshaped = merge.morph(img, img, pts, interpolated, triangles, 1, 0) #second parameter doesn't matter, as we are only taking pixels from the original image
        except np.linalg.LinAlgError:
            print("FAILED")
            continue
        reshaped_faces.append(reshaped)

    average_face = np.zeros(reshaped_faces[0].shape)
    for reshaped in reshaped_faces:
        average_face = np.add(average_face, reshaped)

    average_face = np.divide(average_face, len(reshaped_faces))
    skio.imshow(average_face)
    skio.show()

#    skio.imsave("./average_out.jpg", average_face)
