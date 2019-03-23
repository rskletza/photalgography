import sys
import skimage as sk
from skimage import io, color, draw
import numpy as np
import matplotlib.pyplot as plt
from harris import harris_detector, CorresPoint
import point_detection

if __name__ == "__main__":

    imgpath = sys.argv[1]
    img = sk.img_as_float(color.rgb2gray(io.imread(str(imgpath))))
#    pointlist = harris_detector(imgpath)
#    pointlist = point_detection.filter_points(50, pointlist)
    pointlist = [CorresPoint(0.014591104964992444, 303, 153)]

    x = []
    y = []
    for p in pointlist:
        x.append(p.x)
        y.append(p.y)
        print(p)

    point_detection.extractDescriptors(pointlist, img)

#    for xp, yp in zip(x, y):
#        rr, cc = draw.circle_perimeter(yp, xp, radius=6, shape=img.shape)
#        img[rr, cc] = 1
#    plt.imshow(img)
#    plt.show()
