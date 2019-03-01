import numpy as np
import skimage.io as skio
import skimage as sk
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from matplotlib import path


def parse_pointfile(txt):
    """
    takes a filename and parses points from it. points must either be provided in dlib point file format (as produced by the dlib face_landmark_detection alorithm) or as one point per line, with a space between the coordinates and no other characters 

    returns array with points
    """
    txt = open(txt).read()
    pointlist = []
    #if contains two lines and commas, then is dlib point file
    if "," in txt:
        stringlist = txt.split("\n")
        #discard rectangle (index 0), not needed here
        stringlist = stringlist[1].split(",")
        for i in range(0, len(stringlist)-1, 2):
            if stringlist[i] and stringlist[i+1]:
                pointlist.append([float(stringlist[i]), float(stringlist[i+1])])
    #else is custom point file
    else:
        stringlist = txt.split("\n")
        for string in stringlist:
            if string:
                strings = string.split(" ")
                pointlist.append([float(strings[0]), float(strings[1])])
    return np.array(pointlist)

def add_image_edge_points(pointcloud, img):
    """
    returns a pointcloud with added coordinates for the images edges (8 extra points)
    """
    x_max = img.shape[1]
    y_max = img.shape[0]
    new_points = np.array([[0.0, 0.0], [x_max/2, 0.0], [x_max, 0.0], [0.0, y_max/2], [x_max, y_max/2], [0.0, y_max], [x_max/2, y_max], [x_max, y_max]])
    return np.append(pointcloud, new_points, axis=0)
    

def interpolate_points(pointA, pointB, factor):
    """
    interpolates between two points by factor
    returns interpolated point
    """
    #interpolated = pointA + factor*(pointB - pointA) 
    interpolated = np.add(pointA, np.multiply(factor, np.subtract(pointB, pointA)))
    return interpolated

def interpolate_pointclouds(cloudA, cloudB, factor):
    """
    interpolates for each set of points in a pointcloud
    returns an interpolated pointcloud
    """
    if (len(cloudA) != len(cloudB)):
        raise ValueError('point clouds must be the same length, those given are: ', len(cloudA), len(cloudB))

    if (factor > 1.0 or factor < 0):
        raise ValueError('factor must be between 0 and 1', factor)

    interpolated = []
    for pointA, pointB in zip(cloudA, cloudB):
        interpolated.append(interpolate_points(pointA, pointB, factor))
    return np.array(interpolated)

def reshape_point(point):
    """
    reshapes point to a form that will enable setting up a linear equation
    """
    reshaped = [[point[0], point[1], 1.0, 0, 0, 0], [0, 0, 0, point[0], point[1], 1.0]]
    return reshaped

def to_homog(point):
    """
    converts a point (either as python list or as numpy array) to homogenous coords
    """
    try:
        point.append(1.0)
    except(AttributeError):
        point = np.append(point, 1.0)
    return point

def from_homog(point):
    """
    converts a point in homogenous coords to regular coords
    """
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

    return transform

def calc_transform_all(triangles1, triangles2):
    """
    calculates the affine transformation for a list of triangles
    returns a list of the transformations
    """
    transformations = []
    for tri1, tri2 in zip(triangles1, triangles2):
        transformation = calc_affine_transform(tri1, tri2)
        transformations.append(transformation)
    return np.array(transformations)


def create_triangles(points, indices):
    """
    creates an array of the points belonging to each triangle of a triangulated image
    """
    triangles = []
    for i in range(len(indices)):
       triangle = [points[indices[i][0]], points[indices[i][1]], points[indices[i][2]]]
       triangles.append(triangle)
    return np.array(triangles)

def calc_origin_pixel_color(original_img, pixel_coords, transform_origtoavg):
    """
    applies a transformation to a pixel
    resulting pixel coordinates are used to pick a color from the original image
    uses nearest neighbor approximation
    """
    inverse_transform = np.linalg.inv(transform_origtoavg)
    original_px = from_homog(np.matmul(inverse_transform, to_homog(pixel_coords)))
    #int & round --> nearest neighbor approximation
    orig_x = int(round(original_px[0]))
    orig_y = int(round(original_px[1]))

    #handle edges of image where translated index is out of bounds
    if orig_x >= original_img.shape[1]:
        orig_x = original_img.shape[1]-1
    elif orig_x < 0:
        orig_x = 0
    if orig_y >= original_img.shape[0]:
        orig_y = original_img.shape[0]-1
    elif orig_y < 0:
        orig_y = 0

    px_color = original_img[orig_y, orig_x, :]
    return px_color

def calculate_triangles(img1_pts, img2_pts=[]):
    """
    if given one pointcloud, returns the indices of the Delaunay triangulation
    if given two pointclouds, averages them and then triangulates
    """
    interpolated = img1_pts
    #if we only want triangulation of two pointclouds (average) instead of one
    if len(img2_pts) != 0:
        interpolated = interpolate_pointclouds(img1_pts, img2_pts, 0.5)
    triangles = Delaunay(interpolated).simplices
    return triangles

def morph(img1, img2, img1_pts, img2_pts, triangles, warp_frac, dissolve_frac):
    """
    morphs two images
    first, calculates the average shape using the given points of interest
    warp_frac determines how much of each shape to incorporate (just image1 -> 0)
    then calculates pixel colors based on the pixels of each image
    dissolve_frac determines how much of each image to incorporate (just image1 -> 0)
    """
    interpolated = interpolate_pointclouds(img1_pts, img2_pts, warp_frac)

#    plt.triplot(interpolated[:,0], interpolated[:,1], triangles)
#    plt.plot(interpolated[:,0], interpolated[:,1], 'o')
#    plt.imshow(im1)
#    plt.axis('equal')
#    plt.show()

    triangles_avg = create_triangles(interpolated, triangles)
    triangles_img1 = create_triangles(img1_pts, triangles)
    triangles_img2 = create_triangles(img2_pts, triangles)

    transformations_1toavg = calc_transform_all(triangles_img1, triangles_avg)
    transformations_2toavg = calc_transform_all(triangles_img2, triangles_avg)

    intermed_img = np.zeros_like(img1)
    last_triangle_index = 0 #save last triangle because the next pixel is likely also in it
    transformation_1toavg = transformations_1toavg[last_triangle_index]
    transformation_2toavg = transformations_2toavg[last_triangle_index]

    y_max = intermed_img.shape[0]
    x_max = intermed_img.shape[1]

    #create an array with all possible array indices (cartesian product)
    img_indices = np.array(np.meshgrid(np.arange(0, x_max), np.arange(0, y_max))).T.reshape(-1,2)
    #for each triangle, find all containing pixels and fill according to dissolve_frac
    for t in range(len(triangles_avg)):
        triangle = path.Path(triangles_avg[t])
        transformation_1toavg = transformations_1toavg[t]
        transformation_2toavg = transformations_2toavg[t]

        bool_contained = np.array(triangle.contains_points(img_indices))
        contained = img_indices[bool_contained]
        for i in contained:
            x, y = (i[0], i[1])
            orig_px_1 = np.zeros((3,))
            orig_px_2 = np.zeros((3,))

            if(dissolve_frac != 1): #if we are taking pixels from img1
                orig_px_1 = calc_origin_pixel_color(img1, [x,y], transformation_1toavg)
            if(dissolve_frac != 0): #if we are taking pixels from img2
                orig_px_2 = calc_origin_pixel_color(img2, [x,y], transformation_2toavg)
            #no dissolving (for hybrid images)
            if(dissolve_frac == -1):
                intermed_img[y,x,:] = orig_px_1 + orig_px_2
            else:
                intermed_img[y,x,:] = (1-dissolve_frac) * orig_px_1 + dissolve_frac * orig_px_2
    return intermed_img

