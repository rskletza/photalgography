import numpy as np
import skimage as sk
import skimage.io as skio
import skimage.transform
from scipy.spatial import Delaunay
from numpy import linalg
import matplotlib.pyplot as plt
from matplotlib import path

from splice_image import splice

class PanoSet:
    """
    contains information important for making a panorama, ie the image and the correspondence points between the image on the left and right 
    """
    def __init__(self, img, left_pts, right_pts):
        self.img = img
        self.left_pts = left_pts
        self.right_pts = right_pts

def parse_pointfile(txt):
    """
    takes a filename and parses points from it. 

    returns array with points
    """
    if txt == "dummy":
        return np.array([])
    txt = open(txt).read()
    pointlist = []
    #else is custom point file
    stringlist = txt.split("\n")
    for string in stringlist:
        if string:
            strings = string.split(",")
            pointlist.append([float(strings[0]), float(strings[1])])
    return np.array(pointlist)

def createPanorama(target_set, trans_set, flag):
    """
    takes two PanoSets, transforms one to the same perspective as the other, and combines them
    
    returns a new PanoSet
    """
    if flag == 0: #add to left of base
        target_pts = target_set.left_pts
        dangling_target_pts = target_set.right_pts
        trans_pts = trans_set.right_pts
        dangling_trans_pts = trans_set.left_pts
    else: #add to right of base
        target_pts = target_set.right_pts
        dangling_target_pts = target_set.left_pts
        trans_pts = trans_set.left_pts
        dangling_trans_pts = trans_set.right_pts

    target_img = target_set.img
    trans_img = trans_set.img

#    f, axarr = plt.subplots(1,2)
#    axarr[0].plot(trans_pts[:,0], trans_pts[:,1], 'r+')
#    axarr[0].imshow(trans_img)
#    axarr[0].axis('equal')
#
#    axarr[1].plot(target_pts[:,0], target_pts[:,1], 'r+')
#    axarr[1].imshow(target_img)
#    axarr[1].axis('equal')
#    plt.savefig("correspondences.jpg")
#    plt.show()

    plt.plot(target_pts[:,0], target_pts[:,1], 'b+')
    plt.imshow(target_img)
    plt.axis('off')
    plt.savefig("target.jpg", bbox_inches="tight")
    plt.show()
    
    plt.plot(trans_pts[:,0], trans_pts[:,1], 'b+')
    plt.imshow(trans_img)
    plt.axis('off')
    plt.savefig("trans.jpg", bbox_inches="tight")
    plt.show()
    
    
    H = calcHomography(target_pts, trans_pts)

    intermed_set = applyHomographyToImg(H, trans_set)
    np.save("left_pts", intermed_set.left_pts)
    np.save("right_pts", intermed_set.right_pts)
    np.save("img", intermed_set.img)
#   intermed_set = PanoSet(np.load("img.npy"), np.load("left_pts.npy"), np.load("right_pts.npy"))

    out_set = align(target_set, intermed_set, flag)        
    return out_set

def calcHomography(target_pts, trans_pts):
    """
    calculates the homography that will transform trans_img to the same perspective as ref_img. trans_pts and ref_pts are both arrays of points [x,y] order
    """
    matrix = []
    #print(target_pts)
    #mean = np.mean(target_pts, axis=0)
    #print(mean)
    #sigma = np.std(target_pts, axis=0)
    #print(sigma)
    #target_pts = np.divide(np.subtract(target_pts, mean), sigma)
    #print(target_pts)

    #mean = np.mean(trans_pts, axis=0)
    #sigma = np.std(trans_pts, axis=0)
    #trans_pts = np.divide(np.subtract(trans_pts, mean), sigma)
    for pt, tpt in zip(trans_pts, target_pts):
        rows = createMatrixRows(pt, tpt)
        matrix.append(rows[0])
        matrix.append(rows[1])
    #TODO normalize points

    u, s, v = np.linalg.svd(matrix)
    min_index = np.argmin(s)
    H = np.reshape(v[min_index], (3,3))
    return H

def createMatrixRows(pt, targetpt):
    """
    helper function to create the matrix in order to calculate a homography
    """
    pt_x = pt[0]
    pt_y = pt[1]
    tpt_x = targetpt[0]
    tpt_y = targetpt[1]
    row1 = [-pt_x, -pt_y, -1, 0, 0, 0, pt_x * tpt_x, pt_y * tpt_x, tpt_x]
    row2 = [0, 0, 0, -pt_x, -pt_y, -1, pt_x * tpt_y, pt_y * tpt_y, tpt_y]
    return [row1, row2]

def align(target_set, trans_set, flag):
    """
    takes two sets of images and aligns them based on their points of correspondence
    
    returns a new PanoSet with the correct points of corresponcence
    """
    #differentiation in order to correctly identify the sets of points of corresp.
    if flag == 0: #transl left of base
        print("image is to the left of the base")
        target_pts = target_set.left_pts
        dangling_target_pts = target_set.right_pts
        trans_pts = trans_set.right_pts
        dangling_trans_pts = trans_set.left_pts
    else: #transl right of base
        print("image is to the right of the base")
        target_pts = target_set.right_pts
        dangling_target_pts = target_set.left_pts
        trans_pts = trans_set.left_pts
        dangling_trans_pts = trans_set.right_pts

    target_img = target_set.img
    trans_img = trans_set.img

    #translation from trans_img to target_img (how to move trans_img so that the points of both images overlap exactly)
    transl = np.subtract(target_pts[0],trans_pts[0])
    #calculate new image corners after translation
    trans_corners = [[0,0] + transl, [trans_img.shape[1], 0] + transl, [0, trans_img.shape[0]] + transl, [trans_img.shape[1], trans_img.shape[0]] + transl]
    #calculate the bounding box of the translated image
    trans_min_xy = np.amin(trans_corners, axis=0)
    trans_max_xy = np.amax(trans_corners, axis=0)

    #use target_img as the baseline image
    out_img = target_img
    new_target_pts = target_pts
    new_dangling_target_pts = dangling_target_pts

    #add space around the baseline image (out_img) to accomodate the translated img
    if(trans_min_xy[0] < 0):
        #how much "blackspace" needs to be added on the left
        added_x_left = np.zeros((out_img.shape[0], int(np.ceil(np.abs(trans_min_xy[0]))), 3))
        out_img = np.concatenate((added_x_left, out_img), axis=1)
        new_target_pts += [added_x_left.shape[1], 0]
        new_dangling_target_pts = addToDangling(new_dangling_target_pts, [added_x_left.shape[1], 0])
    if(trans_max_xy[0] > target_img.shape[1]):
        #how much "blackspace" needs to be added on the right
        added_x_right = np.zeros((out_img.shape[0], int(np.ceil(trans_max_xy[0])) - target_img.shape[1], 3))
        out_img = np.concatenate((out_img, added_x_right), axis=1)
    if(trans_min_xy[1] < 0):
        #how much "blackspace" needs to be added on the top
        added_y_top = np.zeros((int(np.ceil(np.abs(trans_min_xy[1]))), out_img.shape[1], 3))
        out_img = np.concatenate((added_y_top, out_img), axis = 0)
        new_target_pts += [0, added_y_top.shape[0]]
        new_dangling_target_pts = addToDangling(new_dangling_target_pts, [0, added_y_top.shape[0]])
    if(trans_max_xy[1] > target_img.shape[0]):
        #how much "blackspace" needs to be added on the bottom
        added_y_bottom = np.zeros((int(np.ceil(trans_max_xy[1]))-target_img.shape[0], out_img.shape[1], 3))
        out_img = np.concatenate((out_img, added_y_bottom), axis = 0)

    #add blackspace to trans_img
    diff_x = np.zeros((trans_img.shape[0], out_img.shape[1] - trans_img.shape[1], 3))
    new_trans_img = np.concatenate((trans_img, diff_x), axis = 1)

    diff_y = np.zeros((out_img.shape[0] - new_trans_img.shape[0], new_trans_img.shape[1], 3))
    new_trans_img = np.concatenate((new_trans_img, diff_y), axis = 0)

    final_transl = np.round(trans_pts[0] - new_target_pts[0]).astype(int)
    new_trans_pts = np.subtract(trans_pts, final_transl)
    new_dangling_trans_pts = addToDangling(dangling_trans_pts, -1 * final_transl)

    #shift the translated image so that the translated image is in the correct place in respect to the baseline image
    translation = skimage.transform.EuclideanTransform(translation=final_transl)
    new_trans_img = skimage.transform.warp(new_trans_img, translation, mode="wrap")

#    print("plot new trans points with new trans img")
#    plt.plot(new_trans_pts[:,0], new_trans_pts[:,1], 'r+')
#    plt.imshow(new_trans_img)
#    plt.axis('equal')
#    plt.show()

    #create a mask for trans img (white for the image, black everywhere else)
    trans_img_mask = (new_trans_img[:,:,:] != (0.0,0.0,0.0)).astype(float)
    out_img_mask = (out_img[:,:,:] != (0.0,0.0,0.0)).astype(float)
    overlapping = np.multiply(trans_img_mask, out_img_mask)
    just_trans = np.subtract(trans_img_mask, overlapping)
    just_out = np.subtract(out_img_mask, overlapping)

#    out_img = np.multiply(just_trans, new_trans_img) + np.multiply(overlapping, np.add(new_trans_img * 0.5, out_img * 0.5)) + np.multiply(just_out, out_img)
#    out_img = np.multiply(just_trans, new_trans_img) + np.multiply(overlapping, out_img) + np.multiply(just_out, out_img)
#    out_img = np.multiply(just_trans, new_trans_img) + np.multiply(overlapping, new_trans_img) + np.multiply(just_out, out_img)

    splice_mask = np.zeros(out_img.shape)
    approx_middle = int(np.mean(new_trans_pts, axis=0)[0]) #np.add(just_out, overlapping)
    #flag = 0 when trans_img is to the left --> want trans_img side to be black (0)
    splice_mask[:,0:approx_middle,:] = [float(flag), float(flag), float(flag)]
    splice_mask[:,approx_middle:,:] = [float(not flag), float(not flag), float(not flag)]
#    skio.imsave("splice_mask.jpg", splice_mask)
    out_img = splice(out_img, new_trans_img, splice_mask, 6)

    if(flag == 0):
        return PanoSet(out_img, new_dangling_trans_pts, new_dangling_target_pts)
    else:
        return PanoSet(out_img, new_dangling_target_pts, new_dangling_trans_pts)

def addToDangling(dangling, vector):
    """
    helper function to handle empty point arrays (dangling points are empty at both "ends" of the panorama images)
    """
    if len(dangling) != 0:
        return np.array(dangling + vector)
    else:
        return np.array([])

def applyHomography(H, points):
    """
    apply a perspective transformation on a set of points
    """
    trans_pts = []
    for pt in points:
        trans_pt = np.matmul(H, to_homog(pt))
        trans_pts.append(np.array(np.round(from_homog(trans_pt))))
    return np.array(trans_pts)

def applyHomographyToImg(H, img_set):
    """
    apply a perspective transformation on a PanoSet
    
    returns a new PanoSet with transformed image and points
    """
    corresp_left = img_set.left_pts
    img = img_set.img
    corresp_right = img_set.right_pts
    points = [[0,0], [img.shape[1], 0], [0, img.shape[0]], [img.shape[1], img.shape[0]]]
    trans_pts = applyHomography(H, points)

    min_xy = np.amin(trans_pts, axis=0)
    max_xy = np.amax(trans_pts, axis=0)
    origin = min_xy.astype(int)
    new_w = int(max_xy[0] - min_xy[0])
    new_h = int(max_xy[1] - min_xy[1])
    out_img = np.zeros((new_h, new_w, 3))
    #TODO limit area by finding triangles
    #TODO parallelize

    transform = lambda p: np.subtract(np.round(from_homog(np.matmul(H, to_homog(p)))).astype(int), origin)
    new_corresp_left = list(map(transform, corresp_left))
    new_corresp_right = list(map(transform, corresp_right))

    print(out_img.shape)
    for y in range(out_img.shape[0]):
        for x in range(out_img.shape[1]):
            out_img[y,x,:] = calc_origin_pixel_color(img, [x+origin[0],y+origin[1]], H)
#    new_corners = np.array([[0,0], [out_img.shape[1], 0], [0, out_img.shape[0]], [out_img.shape[1], out_img.shape[0]]])
#
#    trans_corners = np.subtract(trans_pts, origin).astype(int)
#    pts = np.unique(np.concatenate((trans_corners, new_corners), axis=0), axis=0)
#    triangle_indices = Delaunay(pts).simplices
#    triangles = create_triangles(pts, triangle_indices)
#
#    #create an array with all possible array indices (cartesian product)
#    img_indices = np.array(np.meshgrid(np.arange(0, out_img.shape[1]), np.arange(0, out_img.shape[0]))).T.reshape(-1,2)
#
#    #for each triangle, check if "inside picture"
#    for t in range(len(triangles)):
#        center = np.divide((np.add(np.add(triangles[t][0], triangles[t][1]), triangles[t][2])), 3.0)
#        print(np.array(calc_origin_pixel_color(img, center, H)))
#        if (np.array(calc_origin_pixel_color(img, center, H)).all() == 0):
#            continue
#        print("inside")
#        triangle = path.Path(triangles[t])
#        bool_contained = np.array(triangle.contains_points(img_indices))
#        contained = img_indices[bool_contained]
#        for i in contained:
#            x, y = (i[0], i[1])
#            out_img[y,x,:] = calc_origin_pixel_color(img, [x+origin[0],y+origin[1]], H)

    return PanoSet(out_img, np.array(new_corresp_left), np.array(new_corresp_right))

def create_triangles(points, indices):
    """
    creates an array of the points belonging to each triangle of a triangulated image
    """
    triangles = []
    for i in range(len(indices)):
       triangle = [points[indices[i][0]], points[indices[i][1]], points[indices[i][2]]]
       triangles.append(triangle)
    return np.array(triangles)

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
    if (orig_x >= original_img.shape[1]) or (orig_x < 0) or (orig_y >= original_img.shape[0]) or (orig_y < 0):
#        px_color = [-1, -1, -1]
        px_color = [0, 0, 0]
    else:
        px_color = original_img[orig_y, orig_x, :]

    return px_color
