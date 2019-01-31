import sys
import numpy as np
import skimage as sk
import skimage.io as skio
import math
import os
import simple_cb
import cv2
from skimage.transform import rescale

#find optimal offset in multiple iterations
def multi_iteration_offset(image, base_image, maxoffset):
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
        scaled = sk.transform.rescale(out_image, 1.0/float(n))
        scaled_base = sk.transform.rescale(base_image, 1.0/float(n))

        clipped_scaled = clip_edges(scaled)
        clipped_scaled_base = clip_edges(scaled_base)

        movement = find_offset_by_subtraction(clipped_scaled, clipped_scaled_base, maxoffset)
        
        scaled_movement = np.multiply(movement, n)
        movement_sum += scaled_movement
        out_image = np.roll(out_image, scaled_movement, axis=(0,1))
        
    return movement_sum

##return a fraction (inner n-2/n) of the image for comparison
def clip_edges(image, factor=10):
    width = image.shape[0]
    height = image.shape[1]
    reduced_image = image[int(width/factor):int(width-width/factor), int(height/factor):int(height-height/factor)]
    return reduced_image

#find the optimal offset by calculating the sum of squared differences
#for the interval x,y = [-maxoffset,maxoffset]
def find_offset_by_subtraction(image, base_image, maxoffset):
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
    min_size = 400

    #calculate the maximum scaling factor to have the image be about 400px on the larger edge
    larger_edge = np.argmax(base_image.shape)
    max_scale_power = math.log(min_size/float(base_image.shape[larger_edge]))
    max_scale_power = math.fabs(max_scale_power)
    max_scale_factor = int(math.pow(2, max_scale_power))
    
    scaled = sk.transform.rescale(image, 1.0/float(max_scale_factor))
    scaled_base = sk.transform.rescale(base_image, 1.0/float(max_scale_factor))

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

#convert an index (range [0,maxoffset]) to the corresponding movement (range [-maxoffset, maxoffset])
def to_movement(index, maxoffset):
    return np.subtract(index, maxoffset)

#convert a movement (range [-maxoffset, maxoffset]) to the corresponding index (range [0,maxoffset])
def to_index(movement, maxoffset):
    return np.add(movement, maxoffset)

#calculate optimal movement and apply
def align(image, base_image, maxoffset=15):
    movement = multi_iteration_offset(image, base_image, maxoffset)
    image = np.roll(image, movement, axis=(0,1))
    return image

#calculate optimal movement and rotation and apply
def align_with_rotation(image, base_image, maxoffset=15, maxrot=10, step=0.1):
    rotation = find_rotational_offset_by_subtraction(image, base_image)
    image = sk.transform.rotate(image, rotation)
    movement = multi_iteration_offset(image, base_image, maxoffset)
    image = np.roll(image, movement, axis=(0,1))
    return image

#use edges to find optimal movement and apply
def align_edges(image, base_image, maxoffset=15):
    cv_image = sk.img_as_ubyte(image)
    canny_edge_image = cv2.Canny(cv_image, 50, 200)
    canny_edge_image = sk.img_as_float(canny_edge_image)

    cv_base = sk.img_as_ubyte(base_image)
    canny_edge_base = cv2.Canny(cv_base, 50, 200)
    canny_edge_base = sk.img_as_float(canny_edge_base)

    movement = multi_iteration_offset(canny_edge_image, canny_edge_base, maxoffset)
    image = np.roll(image, movement, axis=(0,1))
    return image

#use gradient to find optimal movement and apply
def align_gradient(image, base_image, maxoffset=15):
    cv_image = sk.img_as_ubyte(image)
    sobel_gradient_image = cv2.Sobel(cv_image, cv2.CV_64F,0,1,ksize=5)
    sobel_gradient_image = sk.img_as_float(sobel_gradient_image)

    cv_base = sk.img_as_ubyte(base_image)
    sobel_gradient_base = cv2.Sobel(cv_base, cv2.CV_64F,0,1,ksize=5)
    sobel_gradient_base = sk.img_as_float(sobel_gradient_base)
    
    movement = multi_iteration_offset(sobel_gradient_image, sobel_gradient_base, maxoffset)
    image = np.roll(image, movement, axis=(0,1))
    return image

#split an image with 3 split color canals (b,g,r) aligned vertically (on top of each other) into three separate images
def split_image(img):
    #calculate height of each sub-image
    height = int(np.floor(img.shape[0] / 3.0))

    #separate color channels
    b = img[:height]
    g = img[height: 2*height]
    r = img[2*height: 3*height]

    return(r,g,b)

#find the borders on the outer edges of an image (eg created by aligning images) and remove them by clipping the image
def remove_border(img, threshold=0.07, area_fraction=10, scale_factor=4):
    scaled_img = sk.transform.rescale(img, 1.0/float(scale_factor))
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
    cv_img = sk.img_as_ubyte(img)
    cv_img = simple_cb.simplest_cb(cv_img, 0.8)
    cv_img = cv2.fastNlMeansDenoisingColored(cv_img, None, 3,3,3,21)
    return sk.img_as_float(cv_img)

for name in sys.argv[1:]:
    img = skio.imread(name)
    
    img = sk.img_as_float(img)

    (r,g,b) = split_image(img)

    #all of the align functions call
    multi_iteration_offset, which in turn calls find_offset_by_subtraction    

    ag = align(g, b)
    ar = align(r, b)

    #uncomment to use edges for alignment (Canny)
#    ag = align_edges(g, b)
#    ar = align_edges(r, b)

    #uncomment to use gradient for alignment (Sobel)
#    ag = align_gradient(g, b)
#    ar = align_gradient(r, b)

    #create color image
    img_out = np.dstack([ar, ag, b])
    img_orig = np.dstack([r,g,b])


    img = remove_border(img_out, 0.07, 10, 4)
    img = image_correct(img_out)
##    skio.imshow(img)
##    skio.show()
#    
    # afficher l'image
#    skio.imshow(img_orig)
#    skio.show()

#    name = os.path.basename(name)
#    name = os.path.splitext(name)[0]
#    skio.imsave("out_" + name + ".jpg", img_out)
