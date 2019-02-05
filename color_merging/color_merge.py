import numpy as np
import skimage as sk
import skimage.io as skio
import math
import simple_cb
import cv2
from skimage.transform import rescale

"""
a number of functions to align, merge and correct split color channel images
"""

def multi_iteration_offset(image, base_image, maxoffset):
    """ returns a movement tuple (x,y)

    find optimal offset in multiple iterations
    """
    min_size = 400
    out_image = image
    movement_sum = (0,0)

    #calculate the maximum scaling factor to have the smallest image be about 400px on the larger edge
    larger_edge = np.argmax(base_image.shape)
    max_scale_power = math.log(400.0/float(base_image.shape[larger_edge]))
    max_scale_power = math.fabs(max_scale_power)
    max_scale_factor = int(math.pow(2, max_scale_power))

    #starting with the largest scale factor (aka the smallest image), sum up the optimal movement
    for n in reversed(range(1,max_scale_factor+1)):
        scaled = sk.transform.rescale(out_image, 1.0/float(n), multichannel=True, anti_aliasing=True)
        scaled_base = sk.transform.rescale(base_image, 1.0/float(n), multichannel=True, anti_aliasing=True)

        clipped_scaled = clip_edges(scaled)
        clipped_scaled_base = clip_edges(scaled_base)

        movement = find_offset_by_subtraction(clipped_scaled, clipped_scaled_base, maxoffset)
        
        scaled_movement = np.multiply(movement, n)
        movement_sum += scaled_movement
        out_image = np.roll(out_image, scaled_movement, axis=(0,1))
        
    return movement_sum

def find_offset_by_subtraction(image, base_image, maxoffset):
    """returns a movement tuple (x,y)

    find the optimal offset by calculating the sum of squared differences for the interval x,y = [-maxoffset,maxoffset]
    """
    #create results array and initialize
    array_width = maxoffset * 2 + 1
    result = np.ones((array_width, array_width))

    #TODO nicely parallelizable
    for j in range(-maxoffset, maxoffset+1):
        for i in range(-maxoffset, maxoffset+1):
            rolled = np.roll(image, (i,j), axis=(0,1))

            res = np.subtract(base_image, rolled)
            res = np.absolute(res)
            res = np.sum(res)
            res = np.power(res, 2)

            result[to_index(j, maxoffset)][to_index(i, maxoffset)] = res

    #get minimum offset
    index_of_min = np.argmin(result)
    index_of_min = np.unravel_index(index_of_min, (array_width, array_width))

    index = (index_of_min[1], index_of_min[0])
    movement = to_movement(index, maxoffset)

    return movement

def find_rotational_offset_by_subtraction(image, base_image, maxrot=10, step=0.1):
    """returns a movement tuple (x,y)

    find the optimal rotational offset by calculating the sum of squared differences for the interval x,y = [-maxoffset,maxoffset]
    """
    min_size = 400

    #calculate the maximum scaling factor to have the image be about 400px on the larger edge
    larger_edge = np.argmax(base_image.shape)
    max_scale_power = math.log(min_size/float(base_image.shape[larger_edge]))
    max_scale_power = math.fabs(max_scale_power)
    max_scale_factor = int(math.pow(2, max_scale_power))
    
    scaled = sk.transform.rescale(image, 1.0/float(max_scale_factor), multichannel=True, anti_aliasing=True)
    scaled_base = sk.transform.rescale(base_image, 1.0/float(max_scale_factor), multichannel=True, anti_aliasing=True)

    array_width = int((maxrot*2)/step)
    result = np.ones(array_width)
    values = list(np.arange(0-maxrot,maxrot,step))
    print(values)
    
    for i in range(len(values)):
        rotated = sk.transform.rotate(scaled, values[i])

        res = np.subtract(scaled_base, rotated)
        res = np.absolute(res)
        res = np.sum(res)
        res = np.power(res, 2)

        result[i] = res

    best_index = (np.argmin(result))
    best_rotation = values[best_index]
    print(best_rotation)

    return best_rotation

def align(image, base_image, maxoffset=15):
    """returns an image array

    calculate optimal movement and apply
    """
    movement = multi_iteration_offset(image, base_image, maxoffset)
    image = np.roll(image, movement, axis=(0,1))
    return image

def align_with_rotation(image, base_image, maxoffset=15, maxrot=10, step=0.1):
    """returns an image array

    calculate optimal movement and rotation and apply
    """
    rotation = find_rotational_offset_by_subtraction(image, base_image)
    image = sk.transform.rotate(image, rotation)
    movement = multi_iteration_offset(image, base_image, maxoffset)
    image = np.roll(image, movement, axis=(0,1))
    return image

def align_edges(image, base_image, maxoffset=15):
    """returns an image array

    use edges to find optimal movement and apply
    """
    cv_image = sk.img_as_ubyte(image)
    canny_edge_image = cv2.Canny(cv_image, 50, 200)
    canny_edge_image = sk.img_as_float(canny_edge_image)

    cv_base = sk.img_as_ubyte(base_image)
    canny_edge_base = cv2.Canny(cv_base, 50, 200)
    canny_edge_base = sk.img_as_float(canny_edge_base)

    movement = multi_iteration_offset(canny_edge_image, canny_edge_base, maxoffset)
    image = np.roll(image, movement, axis=(0,1))
    return image

def align_gradient(image, base_image, maxoffset=15):
    """returns an image array

    use gradient to find optimal movement and apply
    """
    cv_image = sk.img_as_ubyte(image)
    sobel_gradient_image = cv2.Sobel(cv_image, cv2.CV_64F,0,1,ksize=5)
    sobel_gradient_image = sk.img_as_float(sobel_gradient_image)

    cv_base = sk.img_as_ubyte(base_image)
    sobel_gradient_base = cv2.Sobel(cv_base, cv2.CV_64F,0,1,ksize=5)
    sobel_gradient_base = sk.img_as_float(sobel_gradient_base)
    
    movement = multi_iteration_offset(sobel_gradient_image, sobel_gradient_base, maxoffset)
    image = np.roll(image, movement, axis=(0,1))
    return image

def split_image(img):
    """returns a tuple with three image arrays

    splits an image with 3 split color canals (b,g,r) aligned vertically (on top of each other) into three separate images
    """
    #calculate height of each sub-image
    height = int(np.floor(img.shape[0] / 3.0))

    #separate color channels
    b = img[:height]
    g = img[height: 2*height]
    r = img[2*height: 3*height]

    return(r,g,b)

def remove_border(img, threshold=0.07, area_fraction=10, scale_factor=4):
    """returns an image array

    finds the borders on the outer edges of an image (eg created by aligning images) and removes them by clipping the image
    """
    scaled_img = sk.transform.rescale(img, 1.0/float(scale_factor), multichannel=True, anti_aliasing=True)
    search_width = int(scaled_img.shape[1]/area_fraction)
    search_height = int(scaled_img.shape[0]/area_fraction)
    max_width = scaled_img.shape[1]
    max_height = scaled_img.shape[0]

    #top border, left border, bottom border, right border (order is to enable modulo for axis when calculating avg)
    #TODO using list vs using range --> performance loss?
    search_ranges = [list(range(0,search_height)), list(range(0,search_width)), list(reversed(range(max_height-search_height, max_height))), list(reversed(range(max_width-search_width, max_width)))]

    new_borders = [] #to be filled with pixels to crop from each side, in the same order as search_ranges
    for range_index in range(len(search_ranges)):
        border = search_ranges[range_index]
        #this is where the order of the ranges comes in handy
        axis = range_index%2

        #calculate the average of the first line of the border (horizontal or vertical)
        if(axis==0):
            old_avg = np.average(scaled_img[border[0],:], axis=axis) 
        else:
            old_avg = np.average(scaled_img[:,border[0]], axis=0)#axis 0 is used because of the array shape of the extracted column
        
        last_cut = border[0]
        for index in border:
            if(axis==0):
                avg = np.average(scaled_img[index,:], axis=axis)
            else:
                avg = np.average(scaled_img[:,index], axis=0)#axis 0 is used because of the array shape of the extracted column
            diff = np.abs(np.subtract(avg, old_avg))
            if(diff[0] > threshold or diff[1] > threshold or diff[2] > threshold):
                last_cut = index
            old_avg = avg
        new_borders.append(last_cut)

    top_crop = new_borders[0]*scale_factor
    bottom_crop = (max_height-new_borders[2])*scale_factor
    left_crop = new_borders[1]*scale_factor
    right_crop = (max_width-new_borders[3])*scale_factor

    return sk.util.crop(img, ((top_crop,bottom_crop), (left_crop,right_crop), (0,0)))

def image_correct(img):
    """returns an image array

    uses denoising and color correction to make an image look better
    """
    cv_img = sk.img_as_ubyte(img)
    cv_img = simple_cb.simplest_cb(cv_img, 0.8) #from https://gist.github.com/DavidYKay/9dad6c4ab0d8d7dbf3dc
    cv_img = cv2.fastNlMeansDenoisingColored(cv_img, None, 3,3,3,21)
    return sk.img_as_float(cv_img)

#helper functions

def clip_edges(image, factor=10):
    """returns an image array

    clip a fraction (1/n) of the image on each side
    """
    width = image.shape[0]
    height = image.shape[1]
    reduced_image = image[int(width/factor):int(width-width/factor), int(height/factor):int(height-height/factor)]
    return reduced_image

def to_movement(index, maxoffset):
    """returns a tuple

    convert an index (range [0,maxoffset]) to the corresponding movement (range [-maxoffset, maxoffset])
    """
    return np.subtract(index, maxoffset)

def to_index(movement, maxoffset):
    """returns a tuple

    convert a movement (range [-maxoffset, maxoffset]) to the corresponding index (range [0,maxoffset])
    """
    return np.add(movement, maxoffset)

