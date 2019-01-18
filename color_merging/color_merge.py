# TP1, code python pour débuter

# quelques librairies suggérées
# vous pourriez aussi utiliser matplotlib et opencv pour lire, afficher et sauvegarder des images

import sys
import numpy as np
import skimage as sk
import skimage.io as skio

def calc_optimal_offset(maxoffset, base_image, image):
    array_width = maxoffset * 2 + 1
    result = np.ones((array_width, array_width))

    #use numpy iterator nditer
    #nicely parallelizable
    #for each possible offset, calculate quadratic error
    #(error: sum of value differences for each pixel)
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
    #print(result)
    min_index = np.argmin(result)
    min_index = np.unravel_index(min_index, (array_width, array_width))
    min_index = (min_index[1], min_index[0])
    #print("min_index" + str(min_index))
    return min_index

def align(image, base_image, maxoffset=15):
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
    #print(result)
    ##get minimum offset
    #TODO fix inverted index
    offset = np.argmin(result)
    offset = np.unravel_index(offset, (array_width, array_width))
    offset = (offset[1], offset[0])
    print(offset)
    return np.roll(image, offset, axis=(0,1))

def split_image(img):
    # calculer la hauteur de chaque partie (1/3 de la taille de l'image)
    height = int(np.floor(img.shape[0] / 3.0))

    # séparer les canaux de couleur
    b = img[:height]
    g = img[height: 2*height]
    r = img[2*height: 3*height]
    return(r,g,b)

#print(sys.argv[1:])
for name in sys.argv[1:]:
    img = skio.imread(name)
    
    img = sk.img_as_float(img)
    
    (r,g,b) = split_image(img)

    ag = align(g, b)
    ar = align(r, b)

    # créer l'image couleur
    img_out = np.dstack([ar, ag, b])

    # afficher l'image
    skio.imshow(img_out)
    skio.show()

# nom du fichier d'image
# imgname = '00128utif'
#imgname = '01890v.jpg'
#imgname = 'images/00106v.jpg'
#imgname = 'images/00757v.jpg'
#imgname = 'images/00888v.jpg'
#imgname = 'images/00889v.jpg'
#
## lire l'image
#img = skio.imread(imgname)
#
## conversion en double
#img = sk.img_as_float(img)
#    
#(r,g,b) = split_image(img)
#
## aligner les images... c'est ici que vous commencez à coder!
## ces quelques fonctions pourraient vous être utiles:
## np.roll, np.sum, sk.transform.rescale (for multiscale)
#dummy_base = [[1,1,1,0,0], [1,1,1,0,0], [1,1,1,0,0], [0,0,0,0,0], [0,0,0,0,0]]
#dummy_image = [[0,0,0,0,0], [0,0,1,1,1], [0,0,1,1,1], [0,0,1,1,1], [0,0,0,0,0]]
##calc_optimal_offset(1, dummy_base, dummy_image)
#
##print(dummy_base)
##aligned = align(dummy_image, dummy_base)
##print(aligned)
#
#ag = align(g, b)
#ar = align(r, b)
## créer l'image couleur
#img_out = np.dstack([ar, ag, b])

## sauvegarder l'image
#fname = '/out_path/out_fname.jpg'
#skio.imsave(fname, img_out)

## afficher l'image
#skio.imshow(img_out)
#skio.show()
