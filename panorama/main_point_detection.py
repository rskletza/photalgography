import sys
import skimage as sk
from skimage import io, color, draw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from harris import harris_detector, CorresPoint
import point_detection

if __name__ == "__main__":
    imgpath = sys.argv[1]
    imgpath2 = sys.argv[2]
    img1 = sk.img_as_float(color.rgb2gray(io.imread(str(imgpath))))
    img2 = sk.img_as_float(color.rgb2gray(io.imread(str(imgpath2))))
    corrs1, corrs2 = point_detection.find_correspondences(img1, img2)
#    np.save("corrs1", corrs1)
#    np.save("corrs2", corrs2)
    x1 = corrs1[:,0]
    y1 = corrs1[:,1]
    x2 = corrs2[:,0]
    y2 = corrs2[:,1]
    show_img1 = color.gray2rgb(img1)
    show_img2 = color.gray2rgb(img2)
    
    colors = cm.rainbow(np.linspace(0,1,len(corrs1)))[:,:3]
    
    for xp, yp, c in zip(x1, y1, colors):
        rr, cc = draw.circle_perimeter(yp, xp, radius=6, shape=img1.shape)
        show_img1[rr, cc] = c
#    plt.imshow(show_img1)
#    plt.show()

    for xp, yp, c in zip(x2, y2, colors):
        rr, cc = draw.circle_perimeter(yp, xp, radius=6, shape=img2.shape)
        show_img2[rr, cc] = c
#    plt.imshow(show_img2)
#    plt.show()

    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(show_img1)
    axarr[1].imshow(show_img2)
    plt.show()
    

#    x = []
#    y = []
#    for p in pointlist:
#        x.append(p.x)
#        y.append(p.y)
#        print(p)
#
#    for xp, yp in zip(x, y):
#        rr, cc = draw.circle_perimeter(yp, xp, radius=6, shape=img.shape)
#        img[rr, cc] = 1
#    plt.imshow(img)
#    plt.show()
