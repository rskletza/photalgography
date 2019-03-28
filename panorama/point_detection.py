import numpy as np
import numpy.linalg
import numpy.random
import matplotlib.pyplot as plt
from skimage import io as skio
from skimage import color, draw, transform
from sklearn.neighbors import KDTree
import random

import harris
import panorama

def find_correspondences(img1, img2):
#    threshold = 1e-4
#    pointlist1 = harris.harris_detector(img1, threshold) 
#    pointlist2 = harris.harris_detector(img2, threshold)
#    print(pointlist1.shape, pointlist2.shape)
#
#    show_correspondences(img1, img2, np.array([p.coords for p in pointlist1]), np.array([p.coords for p in pointlist2]), corrs=False)
#
##    filter_n = np.min(np.array([pointlist1.shape[0], pointlist2.shape[0]]))
#    filter_n = int(np.min(np.array([pointlist1.shape[0], pointlist2.shape[0]]))/2)
#    print(filter_n)
#    print("filter_points")
#    pointlist1 = filter_points(pointlist1, filter_n)
#    pointlist2 = filter_points(pointlist2, filter_n)
#    print(pointlist1.shape, pointlist2.shape)
#    np.save("pointlist1_filter", pointlist1)
#    np.save("pointlist2_filter", pointlist2)

    pointlist1 = np.load("pointlist1_minfilter.npy")
    pointlist2 = np.load("pointlist2_minfilter.npy")

#    show_correspondences(img1, img2, np.array([p.coords for p in pointlist1]), np.array([p.coords for p in pointlist2]), corrs=False)

#    print("extractDescriptors")
    pointlist1 = extractDescriptors(pointlist1, img1)
    pointlist2 = extractDescriptors(pointlist2, img2)
#
#    print("create descriptor lists")
#    #descriptors1 = np.array([np.array(p.window).flatten() for p in pointlist1])
#    #print(descriptors1[0][0])
#    #print(descriptors1.shape)
#    #print(type(descriptors1[0]))
#    #descriptors2 = np.array([np.array(p.window).flatten() for p in pointlist2])
#
   #this is ridiculous but the above list comprehension is not recognized as the right shape for the KDTree used in match descriptors
#    amount = len(pointlist1)
#    descriptors1 = np.zeros((amount,64))
#    descriptors2 = np.zeros((amount,64))
#    i = 0
#    for p1, p2 in zip(pointlist1, pointlist2):
#        try:
#            descriptors1[i] = p1.window.flatten() 
#            descriptors2[i] = p2.window.flatten()
#        except ValueError:
#            #array is empty, this means that the point was at the edge of the image
#            #no descriptor, so we just leave the "empty" descriptor (an array of zeros)
#            pass
#        i += 1
#
#    print("match descriptors")
#    indices1, indices2 = match_descriptors(descriptors1, descriptors2, 0.4)
#    np.save("indices1", indices1)
#    np.save("indices2", indices2)
##
##    indices1 = np.load("indices1.npy")
##    indices2 = np.load("indices2.npy")
#    
#    coords1 = np.array([pointlist1[i].coords for i in indices1])
#    coords2 = np.array([pointlist2[i].coords for i in indices2])
#
#    show_correspondences(img1, img2, coords1, coords2)
##    return(coords1, coords2)
#
#    matched_points = np.array([(pointlist1[i1], pointlist2[i2]) for i1, i2 in zip(indices1, indices2)])
#    correspondences11, correspondences12 = RANSAC_filter(matched_points)
#    correspondences21, correspondences22 = RANSAC_filter(matched_points)
#    np.save("correspondences11", correspondences11)
#    np.save("correspondences12", correspondences12)
#    np.save("correspondences21", correspondences21)
#    np.save("correspondences22", correspondences22)
    correspondences11 = np.load("correspondences11.npy")
    correspondences12 = np.load("correspondences12.npy")
    correspondences21 = np.load("correspondences21.npy")
    correspondences22 = np.load("correspondences22.npy")

    correspondences1 = correspondences11.tolist()
    correspondences2 = correspondences12.tolist()
    for corr1, corr2 in zip(correspondences21, correspondences22):
        if corr1 in correspondences11:
            #correspondence already exists in list
            continue
        else:
            correspondences1.append(corr1)
            correspondences2.append(corr2)

    return(np.array(correspondences1), np.array(correspondences2))

def filter_points(points, amount):
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

    return np.array(final_set)

def extractDescriptors(pointlist, img):
    for p in pointlist:
        window = img[(p.y-20) : (p.y+20), (p.x-20) : (p.x+20)]

        #window_resized = transform.resize(window, (8,8), anti_aliasing=True)
        #mean = np.mean(window_resized)
        #sigma = np.std(window_resized)
        #window_resized = np.divide(np.subtract(window_resized, mean), sigma)
        #p.window = window_resized

        window_sampled = sample(window, 5)
        mean = np.mean(window_sampled)
        sigma = np.std(window_sampled)
        window_sampled = np.divide(np.subtract(window_sampled, mean), sigma)
        p.window = window_sampled

        #f, axarr = plt.subplots(1,3)
        #axarr[0].imshow(window)
        #axarr[1].imshow(window_resized)
        #axarr[2].imshow(window_sampled)
        #plt.show()

    return pointlist

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

def show_correspondences(img1, img2, indices1, indices2, corrs=True):
    img_cat = np.concatenate((img1, img2), axis=1)
    img1_x = img1.shape[1]
    print(len(indices1))
    for index in indices2:
        index[0] += img1_x

    x1 = indices1[:,0]
    y1 = indices1[:,1]
    for xp, yp in zip(x1, y1):
        rr, cc = draw.circle_perimeter(yp, xp, radius=3, shape=img_cat.shape)
        img_cat[rr, cc] = 1

    x2 = indices2[:,0]
    y2 = indices2[:,1]
    for xp, yp in zip(x2, y2):
        rr, cc = draw.circle_perimeter(yp, xp, radius=3, shape=img_cat.shape)
        img_cat[rr, cc] = 1

    if(corrs):
        for i in range(len(indices1)):
            rr, cc, val = draw.line_aa(y1[i], x1[i], y2[i], x2[i])
            img_cat[rr, cc] = 1

        fig, ax = plt.subplots(figsize = img_cat.shape)
        ax.imshow(img_cat)
        plt.show()
#        img_cat_i = np.copy(img_cat)
#        seq = int(len(indices1))
#        for j in range(0, seq, 10):
#            img_cat_i = np.copy(img_cat)
#            for i in range(j, j+10):
#                rr, cc, val = draw.line_aa(y1[i], x1[i], y2[i], x2[i])
#                img_cat_i[rr, cc] = 1
#
#            fig, ax = plt.subplots(figsize = img_cat_i.shape)
#            ax.imshow(img_cat_i)
#            plt.show()
    else:
        skio.imshow(img_cat)
        skio.show()

def match_descriptors(descriptors1, descriptors2, threshold):
    tree = KDTree(descriptors1)
    dist, ind = tree.query(descriptors2, k=2)
        
    nn2_avg = np.mean(np.array([d[1] for d in dist]))
    #1NN/2NN-avg squared error
    error = np.power(np.divide(dist[:,0], nn2_avg), 2)
    matched = [error < threshold]
    
    descr2_indices = np.array(range(len(descriptors2)))[matched]
    descr1_indices = ind[matched][:,0]
    print(descr2_indices.shape)

    return (descr1_indices, descr2_indices)

def RANSAC_filter(matched_points):
    coords1 = np.array([t[0].coords for t in matched_points])
    coords2 = np.array([t[1].coords for t in matched_points])
    print(coords1.shape, coords2.shape)

    best = {"vote":0, "matched_points":[]}
    while best["vote"] < 4:
        for i in range(10000):
#        point_ind = [np.random.randint(0, len(matched_points)) for i in range(4)]
            point_ind = random.sample(range(0, len(matched_points)), 4)
            selected1 = np.array([coords1[i] for i in point_ind])
            selected2 = np.array([coords2[i] for i in point_ind])

            H = panorama.calcHomography(selected1, selected2)
            trans_coords = panorama.applyHomography(H, coords1)

            ssd = np.power(np.subtract(trans_coords, coords2), 2)
#        print("ssd: " + str(ssd))
#        print(np.min(ssd), np.max(ssd), np.mean(ssd), np.median(ssd))
            filtered = (ssd < 20)
            filtered = np.array([b[0] and b[1] for b in filtered])
            count = np.count_nonzero(filtered)
#        print("vote count: " + str(count))
            if best["vote"] < count:
                best["vote"] = count
                best["matched_indices"] = filtered
                print(count)

    print("best vote:" + str(best["vote"]))

    print(best["matched_indices"])
    matched_indices1 = coords1[best["matched_indices"]]
    matched_indices2 = coords2[best["matched_indices"]]
    print(matched_indices1, matched_indices2)

    return(matched_indices1, matched_indices2) 
