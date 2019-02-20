import sys
import random
import numpy as np
import skimage.io as skio
import skimage as sk
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from matplotlib import path

#if len(sys.argv) != 5:
#    print("please give the images you want to combine. The first image will be used for the low frequencies, the second for the high frequencies (low_frequencies high_frequencies)")
#    sys.exit()

def parse_pointfile(txt):
    stringlist = txt.split("\n")
    pointlist = []
    for string in stringlist:
        if string:
            strings = string.split(" ")
            pointlist.append([float(strings[0]), float(strings[1])])
    return np.array(pointlist)

def add_image_edge_points(pointcloud, img):
    x_max = img.shape[0]
    y_max = img.shape[1]
    new_points = np.array([[0.0, 0.0], [x_max/2, 0.0], [x_max, 0.0], [0.0, y_max/2], [x_max, y_max/2], [0.0, y_max], [x_max/2, y_max], [x_max, y_max]])
    return np.append(pointcloud, new_points, axis=0)
    

def interpolate_points(pointA, pointB, factor):
    #interpolated = pointA + factor*(pointB - pointA) 
    interpolated = np.add(pointA, np.multiply(factor, np.subtract(pointB, pointA)))
    return interpolated

def interpolate_pointclouds(cloudA, cloudB, factor):
    if (len(cloudA) != len(cloudB)):
        raise ValueError('point clouds must be the same length, those given are: ', len(cloudA), len(cloudB))

    if (factor > 1.0 or factor < 0):
        raise ValueError('factor must be between 0 and 1', factor)

    interpolated = []
    for pointA, pointB in zip(cloudA, cloudB):
        interpolated.append(interpolate_points(pointA, pointB, factor))
    return np.array(interpolated)

def reshape_point(point):
    reshaped = [[point[0], point[1], 1.0, 0, 0, 0], [0, 0, 0, point[0], point[1], 1.0]]
    return reshaped

def to_homog(point):
    try:
        point.append(1.0)
    except(AttributeError):
        point = np.append(point, 1.0)
    return point

def from_homog(point):
    return [point[0]/point[2], point[1]/point[2]]

def calc_affine_transform(triangle1, triangle2):
    dependents = []
    for point in triangle1:
        dependents += reshape_point(point)
    dependents = np.array(dependents)
    triangle2 = np.array(triangle2)

    transform = np.linalg.solve(dependents, np.ndarray.flatten(triangle2))
    transform = np.append(transform, [0.0, 0.0, 1.0])
    transform = np.reshape(transform, (3,3))

    #print(np.matmul(transform, to_homog(triangle1[0])))
    return transform

def create_triangles(points, indices):
    triangles = []
    for i in range(len(indices)):
       triangle = [points[indices[i][0]], points[indices[i][1]], points[indices[i][2]]]
       triangles.append(triangle)
    print(triangles[0])
    return np.array(triangles)

def calc_transform_all(triangles1, triangles2):
    transformations = []
    for tri1, tri2 in zip(triangles1, triangles2):
        transformation = calc_affine_transform(tri1, tri2)
        transformations.append(transformation)
    return np.array(transformations)

def calc_origin_pixel_color(original_img, pixel_coords, transform_origtoavg):
    inverse_transform = np.linalg.inv(transform_origtoavg)
    original_px = from_homog(np.matmul(inverse_transform, to_homog(pixel_coords)))
    px_color = original_img[int(np.round(original_px[1])), int(np.round(original_px[0])), :]
    return px_color


def morph(img1, img2, img1_pts, img2_pts, tri, warp_frac, dissolve_frac):
    interpolated = interpolate_pointclouds(img1_pts, img2_pts, warp_frac)
    triangulated = Delaunay(interpolated)

##plt.triplot(interpolated[:,0], interpolated[:,1], triangulated.simplices)
##plt.plot(interpolated[:,0], interpolated[:,1], 'o')
#    plt.triplot(points1[:,0], points1[:,1], triangulated.simplices)
#    plt.plot(points1[:,0], points1[:,1], 'o')
#    plt.imshow(im1)
#    plt.axis('equal')
##plt.gca().invert_yaxis()
#    plt.show()

    triangles_avg = create_triangles(interpolated, triangulated.simplices)
    triangles_img1 = create_triangles(img1_pts, triangulated.simplices)
    triangles_img2 = create_triangles(img2_pts, triangulated.simplices)

    transformations_1toavg = calc_transform_all(triangles_img1, triangles_avg)
    transformations_2toavg = calc_transform_all(triangles_img2, triangles_avg)
    print(transformations_1toavg.shape)

    intermed_img = np.zeros_like(img1)
    last_triangle_index = 0 #save last triangle because the next pixel is likely also in it
    transformation_1toavg = transformations_1toavg[last_triangle_index]
    transformation_2toavg = transformations_2toavg[last_triangle_index]

    colors = [[255,0,0], [0,255,0], [0,0,255], [255,255,0], [0,255,255], [255,0,255]]
    color = [255,0,0]
    for y in range(intermed_img.shape[1]):
        for x in range(intermed_img.shape[0]):
            pixel = intermed_img[y,x,:]
            if not (path.Path(triangles_avg[last_triangle_index]).contains_point([x,y])):
                for t in range(len(triangles_avg)):
                    triangle_path = path.Path(triangles_avg[t])
                    if triangle_path.contains_point([x,y]):
                        color = colors[t%len(colors)]
                        transformation_1toavg = transformations_1toavg[t]
                        transformation_2toavg = transformations_2toavg[t]
                        last_triangle_index = t
            intermed_img[y,x,:] = color
            intermed_img[y,x,:] = calc_origin_pixel_color(img1, [x,y], transformation_1toavg)

    skio.imshow(intermed_img)
    skio.show()


im1 = skio.imread("./faces/19-Rosalie.jpg")
im2 = skio.imread("./faces/22-Eric.jpg")

points1 = parse_pointfile(open("./points/19-Rosalie.txt").read())
points1 = add_image_edge_points(points1, im1)
points2 = parse_pointfile(open("./points/22-Eric.txt").read())
points2 = add_image_edge_points(points2, im2)

im1 = sk.img_as_float(im1)
im2 = sk.img_as_float(im2)

interpolated = interpolate_pointclouds(points1, points2, 0.5)
triangulated = Delaunay(interpolated)

morphed = morph(im1, im2, points1, points2, triangulated, 0.5, 0.5)

#triangle1 = np.array([ points1[triangulated.simplices[0][0]], points1[triangulated.simplices[0][1]], points1[triangulated.simplices[0][2]]])
#triangle2 = np.array([ interpolated[triangulated.simplices[0][0]], interpolated[triangulated.simplices[0][1]], interpolated[triangulated.simplices[0][2]]])
#print(triangle1.shape, triangle2.shape)
#calc_affine_transform(triangle1, triangle2)


#skio.imshow(im12)
#skio.show()
#uncomment to save image
#skio.imsave("", im12)
