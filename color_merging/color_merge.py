# TP1, code python pour débuter

# quelques librairies suggérées
# vous pourriez aussi utiliser matplotlib et opencv pour lire, afficher et sauvegarder des images

import sys
import numpy as np
import skimage as sk
import skimage.io as skio
from skimage.transform import rescale

def multi_iteration_offset(image, base_image, maxoffset):
    out_image = image
    max_scale_factor = 4

#    if(base_image.shape[0]*base_image.shape[1] > 160000): #larger than ~400x400
        #TODO more sophisticated scaling --> scale to approx 1600000 area
#        scale_factor = int(np.max(base_image.shape)/400)

    for n in reversed(range(1,max_scale_factor+1)):
        print(n)
        scaled = sk.transform.rescale(image, (1/n, 1/n))
        scaled_base = sk.transform.rescale(base_image, 1.0/float(n))
        movement = find_offset_by_subtraction(scaled, scaled_base, maxoffset)
        print(movement)
        
        scaled_movement = np.multiply(movement, n)
        print(scaled_movement)
        out_image = np.roll(image, scaled_movement, axis=(0,1))
        
        skio.imshow(scaled)
        skio.show()


        
    return(0,0)

def clip_edges(image, factor=10):
    ##use a fraction (inner 2/n) of the image for comparison
    print(image.shape)
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
    for j in range(array_width):
        for i in range(array_width):
            #print((i,j))
            rolled = np.roll(image, (i,j), axis=(0,1))

            #print(rolled)

            res = np.subtract(base_image, rolled)
            res = np.absolute(res)
            res = np.sum(res)
            res = np.power(res, 2)
            #print(res)
            result[j][i] = res
    ##get minimum offset
    index_of_min = np.argmin(result)
    index_of_min = np.unravel_index(index_of_min, (array_width, array_width))
    #print(result[index_of_min])
    #print(np.average(result, axis=(0,1)))

    #translate index into movement
    movement = (index_of_min[1], index_of_min[0])
    return movement

def align(image, base_image, maxoffset=15):

    reduced_image = clip_edges(image)
    reduced_base = clip_edges(base_image)
    #movement = multi_iteration_offset(reduced_image, reduced_base, maxoffset)
    movement = find_offset_by_subtraction(reduced_image, reduced_base, maxoffset)
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
#dummy_image = [0,0,0,0,0, 0,0,1,1,1, 0,0,1,1,1, 0,0,1,1,1, 0,0,0,0,0]
#dummy_image = np.reshape(dummy_image, (5,5))
#
#print(dummy_base)
#aligned = align(dummy_image, dummy_base, 5)
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
