import sys
import os
import color_merge as cm
import skimage as sk
import skimage.io as skio
import numpy as np

#loop through all the files given as arguments
for name in sys.argv[1:]:
    img = skio.imread(name)
    
    img = sk.img_as_float(img)

    (r,g,b) = cm.split_image(img)

    #all of the align functions call cm.multi_iteration_offset, which in turn calls cm.find_offset_by_subtraction    
    ag = cm.align(g, b)
    ar = cm.align(r, b)

    #uncomment to use edges for alignment (Canny)
#    ag = cm.align_edges(g, b)
#    ar = cm.align_edges(r, b)

    #uncomment to use gradient for alignment (Sobel)
#    ag = cm.align_gradient(g, b)
#    ar = cm.align_gradient(r, b)

    #create color image
    img_out = np.dstack([ar, ag, b])
    img_orig = np.dstack([r,g,b])

    img_out = cm.remove_border(img_out, 0.07, 10, 4)
    img_out = cm.image_correct(img_out)

    #uncomment to view original image
#    skio.imshow(img_orig)
#    skio.show()

    skio.imshow(img_out)
    skio.show()
    
    #uncomment to save image
#    name = os.path.basename(name)
#    name = os.path.splitext(name)[0]
#    skio.imsave("out_" + name + ".jpg", img_out)
