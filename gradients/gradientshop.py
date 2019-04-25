import numpy as np
import numpy.linalg
import numpy.random
import matplotlib.pyplot as plt
import cv2
import time
from scipy.signal import convolve2d as conv2
from scipy.optimize import minimize
from skimage import io, filters, draw, transform
from matplotlib import pyplot as plt

def calc_saliency(img):
    """
    calculate the saliency of the local gradients: long, coherent edges are perceptually more important to human perception. 
    this function estimates the length of continuous gradients and returns an image with two channels e_l (length of edge containing certain pixel) and e_o(orientation of edge at this pixel)
    """
    def get_q(p_index, direction):
        """
        get the exact index of q, calculated by moving sqrt(2) along the edge in a specified direction
        """
        theta = orient[p_index[1], p_index[0]] + np.pi/2.0 + direction*np.pi
#        print(np.rad2deg( orient[p_index[1], p_index[0]]))

#        unit_vector = np.divide(np.array([np.cos(theta), np.sin(theta)]), magn[p_index[1], p_index[0]])
#        vector = unit_vector * np.sqrt(2)
#        q_exact = p_index + vector
        q_exact = np.array([ p_index[0] + np.cos(theta) * np.sqrt(2), p_index[1] + np.sin(theta) * np.sqrt(2) ])
#        print("q: " + str(q_exact))
        return q_exact

    def interpolate_q(q, array):
        pixels = np.array([ np.floor(q), np.ceil(q), np.array([np.floor(q[0]), np.ceil(q[1])]), np.array([np.ceil(q[0]), np.floor(q[1])]) ]).astype(int)
        interpolated = 0 # array[pixels[0][1], pixels[0][0]] + array[pixels[1]]
        samples = 0
        for p in pixels:
            try:
                interpolated += array[p[1], p[0]]
                samples += 1
            except IndexError:
                continue
        if samples == 0:
            return 0
        else:
            interpolated /= samples
            return interpolated
#        q = np.round(q).astype(int)
#        x = q[0]
#        y = q[1]
#        print(x, y)
#        print(array.shape)
#        if x >= array.shape[1]:
#            x = array.shape[1]-1
#        if y >= array.shape[0]:
#            y = array.shape[0]-1
#        return array[y, x]

    def w_theta(p, q):
        """
        measures similarity of the local edge orientations (using gradient orientation, but that should not make a difference)
        """
        q_theta = interpolate_q(q, orient)
        p_theta = orient[p[1], p[0]]
        weight = np.exp( -1 * np.power((p_theta -  q_theta), 2) / (2 * np.pi/5.0))
        return weight

    def w_alpha(q):
        #TODO don't know what this is supposed to do
        return 1

    #TODO steerable filters instead of finite difference?
    print(img.shape)
    img_blurred = filters.gaussian(img, sigma=1)
    gx = conv2(img_blurred, np.array([[-1, 0, 1]]), 'same', 'symm')  # take x derivative
    gy = conv2(img_blurred, np.transpose(np.array([[-1, 0, 1]])), 'same', 'symm')  # take y derivative

    #calculate gradient orientation in rad
#    orient = np.arctan(np.divide(gy, gx))
#    magn = np.sqrt(np.add(np.power(gy, 2), np.power(gx, 2)))
    magn, orient = cv2.cartToPolar(gx, gy, angleInDegrees=False)

    normalized_magn = block_normalize(magn)
#    f, axarr = plt.subplots(1,2)
#    axarr[0].imshow(gx + gy + 0.5)
#    axarr[1].imshow(normalized_magn)
#    plt.show()

    m0_prev = np.zeros((img.shape))
    m1_prev = np.zeros((img.shape))
    m0_new = np.zeros((img.shape))
    m1_new = np.zeros((img.shape))
    scale = 20
    test_img = transform.rescale(normalized_magn, scale)
    start = time.time()
    for i in range(40):
        for y in range(0, img.shape[0]):
            for x in range(0, img.shape[1]):
                q = get_q([x,y], 0)
#                rr, cc = draw.line(y*scale, x*scale, np.round(q[1]).astype(int)*scale, np.round(q[0]).astype(int)*scale)
#                try:
#                    test_img[rr, cc] = 1
#                except IndexError:
#                    pass
#                io.imshow(test_img)
#                io.show()
                m0_new[y,x] = w_alpha(q) * w_theta([x,y], q) * (interpolate_q(q, normalized_magn) + interpolate_q(q, m0_prev))
                q = get_q([x,y], 1)
                m1_new[y,x] = w_alpha(q) * w_theta([x,y], q) * (interpolate_q(q, normalized_magn) + interpolate_q(q, m1_prev))
#        io.imshow(test_img)
#        io.show()
        m0_prev = m0_new
        m1_prev = m1_new
#        print(m0_prev)
#    io.imshow(m0_prev)
#    io.show()
#    io.imshow(m1_prev)
#    io.show()
    end = time.time()
    print(end - start)

    lengths = m0_new + m1_new + normalized_magn
    np.save("lengths-out.npy", lengths)
#    lengths = np.load("./lengths-tiger-10it.npy")

    sx = np.power(np.cos(orient), 2) * lengths * gx
    sy = np.power(np.sin(orient), 2) * lengths * gy

    f, axarr = plt.subplots(1,5)
    axarr[0].imshow(img)
    axarr[1].imshow(gx + gy + 0.5)
    axarr[2].imshow(normalized_magn)
    axarr[3].imshow(lengths)
    axarr[4].imshow(np.abs(sx + sy))
    plt.show()

    return sx, sy

def block_normalize(img):
    """
    normalize values locally in blocks of 5x5 (subtract by mean and divide by standard deviation)
    """
    mean_filter = np.full((5,5), 1/25.0)
    epsilon = 1e-01

    blocked_mean = conv2(img, mean_filter, mode = "same", boundary = "symm")

    blocked_std = np.power((img - blocked_mean), 2)
    blocked_std = conv2(blocked_std, mean_filter, mode = "same", boundary = "symm")
    blocked_std = np.sqrt(blocked_std)
    
    normalized_img = np.divide( (img - blocked_mean), (blocked_std + epsilon) )
    #TODO question! centered around 0???
#    min_value = np.min(normalized_img)
#    if(min_value < 0):
#        normalized_img = normalized_img + np.absolute(np.min(normalized_img))
    return np.abs(normalized_img)
    
def constrained_filter(d, g, w, u):
    """
    d: intensity constraints (1 channel)
    g: gradient constraints (h and v, 2 channels)
    w: weights of d and g (3 channels)
    u: input image (intensities, h gradients, v gradients: 3 channels)
    """
    def loss(f):
        f = np.reshape(f, (u.shape))
        fgx = conv2(f, np.array([[-1, 0, 1]]), 'same')  # take x derivative
        fgy = conv2(f, np.transpose(np.array([[-1, 0, 1]])), 'same')  # take y derivative
        Ed = w[:,:,0] * np.power(f - d, 2)
        Eg = w[:,:,1] * np.power(fgx-g[:,:,0], 2) + w[:,:,2] * np.power(fgy-g[:,:,1], 2)
        return np.sum(Ed + Eg)

    res = minimize(loss, u, method="CG")    
    np.save("result.npy", res.x)
    np.save("jacobien.npy", res.jac)
    return res.x

def basic_sharpening(img):
    img_blurred = filters.gaussian(img, sigma=1)
    gx = conv2(img_blurred, np.array([[-1, 0, 1]]), 'same')  # take x derivative
    gy = conv2(img_blurred, np.transpose(np.array([[-1, 0, 1]])), 'same')  # take y derivative
    cs = 10000
    c1 = 1
    d = img
    g = np.dstack((gx * cs, gy * cs))
    print(gx[0,0])
    print(g[0,0,0])
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(gx)
    axarr[1].imshow(g[:,:,0])
    plt.show()
    
    w = np.dstack((np.full(img.shape, c1), np.ones(img.shape), np.ones(img.shape)))
    u = img#np.dstack((img, gx, gy))
    #start = time.time()
    #res = constrained_filter(d, g, w, u)
    #end = time.time()
    #print(str((end-start)/60) + "min")

    res = np.load("result.npy")
    res = np.reshape(res, (u.shape))
    print(np.sum(res - img))
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(img)
    axarr[1].imshow(res)
    plt.show()



