import numpy as np
import numpy.linalg
from skimage import io as skio

def filter_points(amount, points):
    final_set = []
    threshold = 0.9

    global_max = np.max(points)
    final_set.append(global_max)

    rvals = []
    i = 0
    for pi in points:
        thresh_pts = [np.linalg.norm(np.subtract(pi.coords, pj.coords)) for pj in points if (pi.v < threshold * pj.v)]
        if len(thresh_pts) == 0:
            ri = 0
        else:
            ri = np.min(thresh_pts)
        rvals.append((ri,i))
        i += 1

    sorted_list = sorted(rvals, key=lambda t:t[0])

    highest_n = sorted_list[-amount:]
    for r, i in highest_n:
        final_set.append(points[i])

    return final_set

def extractDescriptors(pointlist, img):
    print(img.shape)
    for p in pointlist:
        window = img[(p.y-20) : (p.y+20), (p.x-20) : (p.x+20)]
        skio.imshow(window)
        skio.show()

        window = sample(window, 5)
        mean = np.mean(window)
        sigma = np.std(window)
        window = np.divide(np.subtract(window, mean), sigma)

        print(window)
        skio.imshow(window)
        skio.show()

def sample(img, n):
    """
    returns a reduced image (achieved by sampling every nth pixel) 
    """
    x_indices_set = set(range(img.shape[0]))
    x_indices_to_keep = set(range(1, img.shape[0],n))
    x_indices_to_delete = list(x_indices_set - x_indices_to_keep)
    img = np.delete(img,x_indices_to_delete, axis=0)

    y_indices_set = set(range(img.shape[1]))
    y_indices_to_keep = set(range(1, img.shape[1],n))
    y_indices_to_delete = list(y_indices_set - y_indices_to_keep)
    img = np.delete(img, y_indices_to_delete, axis=1)
    return img
