import sys
import numpy as np
import skimage as sk
import skimage.io as skio
import math
import os
from skimage.transform import rescale

def multi_iteration_offset(image, base_image, maxoffset):
    min_size = 400
    out_image = image
    movement_sum = (0,0)

    #calculate the maximum scaling factor to have the smallest image be about 400px on the larger edge
    larger_edge = np.argmax(base_image.shape)
    max_scale_power = math.log(400.0/float(base_image.shape[larger_edge]))
    max_scale_power = math.fabs(max_scale_power)
    max_scale_factor = int(math.pow(2, max_scale_power))
    #print(max_scale_factor)

    for n in reversed(range(1,max_scale_factor+1)):
        print(n)
        scaled = sk.transform.rescale(out_image, 1.0/float(n))
        scaled_base = sk.transform.rescale(base_image, 1.0/float(n))
        #print("original: " + str(image.shape) + ", scaled: " + str(scaled.shape))

        scaled = clip_edges(scaled)
        scaled_base = clip_edges(scaled_base)

        movement = find_offset_by_subtraction(scaled, scaled_base, maxoffset)
        
        print("movement: " + str(movement))
        scaled_movement = np.multiply(movement, n)
        print("scaled_movement: " + str(scaled_movement))
        movement_sum += scaled_movement
        print("movement_sum: " + str(movement_sum))
        out_image = np.roll(out_image, scaled_movement, axis=(0,1))
        
    return movement_sum

def clip_edges(image, factor=10):
    ##use a fraction (inner n-2/n) of the image for comparison
    width = image.shape[0]
    height = image.shape[1]
    reduced_image = image[int(width/factor):int(width-width/factor), int(height/factor):int(height-height/factor)]
#    skio.imshow(reduced_image)
#    skio.show()
    return reduced_image

def find_offset_by_subtraction(image, base_image, maxoffset):
    #create results array
    array_width = maxoffset * 2 + 1
    result = np.ones((array_width, array_width))

    ##for each possible offset, calculate quadratic error
    ##(error: sum of value differences for each pixel)
    #TODO use numpy iterator nditer
    #TODO nicely parallelizable
#    for j in range(array_width):
#        for i in range(array_width):
    for j in range(-maxoffset, maxoffset+1):
        for i in range(-maxoffset, maxoffset+1):
            #print((i,j))
            rolled = np.roll(image, (i,j), axis=(0,1))

            #print(rolled)

            res = np.subtract(base_image, rolled)
            res = np.absolute(res)
            res = np.sum(res)
            res = np.power(res, 2)
            #print(res)
            result[to_index(j, maxoffset)][to_index(i, maxoffset)] = res
    ##get minimum offset
    index_of_min = np.argmin(result)
    index_of_min = np.unravel_index(index_of_min, (array_width, array_width))
    print(result[index_of_min])
    print(np.average(result, axis=(0,1)))

    #translate index into movement
    index = (index_of_min[1], index_of_min[0])
    movement = to_movement(index, maxoffset)
    #print(movement)
    return movement

def to_movement(index, maxoffset):
    #return index
    return np.subtract(index, maxoffset)

def to_index(movement, maxoffset):
    return np.add(movement, maxoffset)

def align(image, base_image, maxoffset=15):

#    reduced_image = clip_edges(image)
#    reduced_base = clip_edges(base_image)
    movement = multi_iteration_offset(image, base_image, maxoffset)
#    movement = find_offset_by_subtraction(image, base_image, maxoffset)
    return np.roll(image, movement, axis=(0,1))

def split_image(img):
    # calculer la hauteur de chaque partie (1/3 de la taille de l'image)
    height = int(np.floor(img.shape[0] / 3.0))

    # séparer les canaux de couleur
    b = img[:height]
    g = img[height: 2*height]
    r = img[2*height: 3*height]
    return(r,g,b)

#dummy_base = [1,1,1,0,0, 1,1,1,0,0, 1,1,1,0,0, 0,0,0,0,0, 0,0,0,0,0]
#dummy_base = np.reshape(dummy_base, (5,5))
##dummy_image = [0,0,0,0,0, 0,0,1,1,1, 0,0,1,1,1, 0,0,1,1,1, 0,0,0,0,0]
##dummy_image = [1,1,0,0,1, 1,1,0,0,1, 0,0,0,0,0, 0,0,0,0,0, 1,1,0,0,1]
#dummy_image = [0,1,1,1,0, 0,1,1,1,0, 0,1,1,1,0, 0,0,0,0,0, 0,0,0,0,0]
#dummy_image = np.reshape(dummy_image, (5,5))
##
#print(dummy_base)
#aligned = align(dummy_image, dummy_base, 4)
#print(aligned)

for name in sys.argv[1:]:
    img = skio.imread(name)
    
    img = sk.img_as_float(img)
    
    (r,g,b) = split_image(img)

    ag = align(g, b)
    ar = align(r, b)

    # créer l'image couleur
    img_out = np.dstack([ar, ag, b])
    img_orig = np.dstack([r,g,b])

    # afficher l'image
    skio.imshow(img_orig)
    skio.show()
    skio.imshow(img_out)
    skio.show()

    name = os.path.basename(name)
    name = os.path.splitext(name)[0]
    skio.imsave("personal_out/out_" + name + ".jpg", img_out)
