import numpy as np
import numpy.linalg
import numpy.random
import matplotlib.pyplot as plt
import cv2
import time
from scipy.signal import convolve2d as conv2
from scipy.optimize import minimize, least_squares
from skimage import io, filters, draw, transform, exposure
from matplotlib import pyplot as plt

def calc_saliency(img):
    """
    calculate the saliency of the local gradients: long, coherent edges are perceptually more important to human perception. 
    this function estimates the length of continuous gradients and returns an image with two channels e_l (length of edge containing certain pixel) and e_o(orientation of edge at this pixel)
    """
    def get_q(p_index, direction):
        """
        get the exact index of q, calculated by moving sqrt(2) along the edge in a specified direction (either in the direction of the edge (orientation + pi/2), or in the other direction of the edge (orientation + p/2 + pi))
        """
        edge_dir = orient[p_index[1], p_index[0]] + np.pi/2.0 + direction*np.pi
        q_exact = np.array([ p_index[0] + np.cos(edge_dir) * np.sqrt(2), p_index[1] + np.sin(edge_dir) * np.sqrt(2) ])
        return q_exact

    def get_qs(direction):
        """
        get the exact index of q, calculated by moving sqrt(2) along the edge in a specified direction (either in the direction of the edge (orientation + pi/2), or in the other direction of the edge (orientation + p/2 + pi))
        """
        edge_dirs = orient + np.pi/2.0 + direction*np.pi
        positions = np.indices((img.shape))
        q_vector_x = np.cos(edge_dirs) * np.sqrt(2)
        q_vector_y = np.sin(edge_dirs) * np.sqrt(2)
        q_vectors = np.dstack((q_vector_y, q_vector_x))
        #creates an array with the position (index) at each index
        indices = np.swapaxes(np.swapaxes(np.indices((orient.shape)), 0, 1), 1, 2)
        qs_flippedindex = indices + q_vectors
        qs_indices = np.flip(qs_flippedindex, axis = 2)
        return qs_indices

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

    def interpolate_qs(qs, array):
        interpol_array = np.zeros((img.shape))
        for y in range(0, qs.shape[0]):
            for x in range(0, qs.shape[1]):
                q = qs[y,x]
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
                    interpolated = 0
                else:
                    interpolated /= samples
                interpol_array[y,x] = interpolated
        return interpol_array

    def w_theta(p, q):
        """
        measures similarity of the local edge orientations (using gradient orientation, but that should not make a difference)
        """
        q_theta = interpolate_q(q, orient)
        p_theta = orient[p[1], p[0]]
        weight = np.exp( -1 * np.power((p_theta -  q_theta), 2) / (2 * np.pi/5.0))
        return weight

    def w_theta_vectorized(qs):
        """
        measures similarity of the local edge orientations (using gradient orientation, but that should not make a difference)
        """
        p_theta = orient
        q_theta = interpolate_qs(qs, orient)
#        q_theta = interpolate_q(qs, orient)
        weight = np.exp( -1 * np.power((p_theta -  q_theta), 2) / (2 * np.pi/5.0))
        return weight

    def w_alpha_vectorized(qs):
        #TODO don't know what this is supposed to do
        return np.ones(img.shape)
    
    def w_alpha(q):
        #TODO don't know what this is supposed to do
        return 1

    #TODO steerable filters instead of finite difference?
    print(img.shape)
    filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gx = conv2(img, filter, 'same', 'symm')  # take x derivative
    gy = conv2(img, np.transpose(filter), 'same', 'symm')  # take y derivative

    #calculate gradient orientation in rad
#    orient = np.arctan(np.divide(gy, gx))
#    magn = np.sqrt(np.add(np.power(gy, 2), np.power(gx, 2)))
    magn, orient = cv2.cartToPolar(gx, gy, angleInDegrees=False)

    normalized_magn = block_normalize(magn)
#    f, axarr = plt.subplots(1,2)
#    axarr[0].imshow(gx + gy + 0.5)
#    axarr[1].imshow(normalized_magn)
#    plt.show()

#    m0_prev = np.zeros((img.shape))
#    m1_prev = np.zeros((img.shape))
#    m0_new = np.zeros((img.shape))
#    m1_new = np.zeros((img.shape))
#    scale = 20
#    test_img = transform.rescale(normalized_magn, scale)
#    start = time.time()
#    m0_qs = get_qs(0)
#    m1_qs = get_qs(1)
#    w_alphas_0 = w_alpha_vectorized(m0_qs)
#    w_alphas_1 = w_alpha_vectorized(m1_qs)
#    w_thetas_0 = w_theta_vectorized(m0_qs)
#    w_thetas_1 = w_theta_vectorized(m1_qs)
#    interpol_magn_0 = interpolate_qs(m0_qs, normalized_magn)
#    interpol_magn_1 = interpolate_qs(m1_qs, normalized_magn)
#    for i in range(40):
#        m0_new = w_alphas_0 * w_thetas_0 * (interpol_magn_0 + interpolate_qs(m0_qs, m0_prev))
#        m1_new = w_alphas_1 * w_thetas_1 * (interpol_magn_1 + interpolate_qs(m1_qs, m1_prev))
#        m0_prev = m0_new
#        m1_prev = m1_new
#
#    end = time.time()
#    print("execution time: " + str(end - start) + "s")
#
#    lengths = m0_new + m1_new + normalized_magn
#    np.save("lengths-pixi.npy", lengths)
    lengths = np.load("./lengths-pixi.npy")

    sx = np.power(np.cos(orient), 2) * lengths * gx
    sy = np.power(np.sin(orient), 2) * lengths * gy

    f, axarr = plt.subplots(1,3)
#    axarr[0].imshow(img)
    axarr[0].imshow(gx + gy + 0.5)
#    axarr[2].imshow(normalized_magn)
    axarr[1].imshow(lengths)
    axarr[2].imshow(sx + sy)
    plt.show()

#    io.imsave("0_grayscale.jpg", img)
#    io.imsave("1_gradients.jpg", exposure.rescale_intensity(gx + gy + 0.5))
#    io.imsave("2_normalized_gradients.jpg", exposure.rescale_intensity(normalized_magn))
#    io.imsave("3_lengths.jpg", exposure.rescale_intensity(lengths))
#    io.imsave("4_final.jpg", exposure.rescale_intensity(np.abs(sx + sy)))

    el = lengths
    eo = orient

    return sx, sy, el, eo

def get_gradients(img):
    filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gx = conv2(img, filter, 'same', 'symm')  # take x derivative
    gy = conv2(img, np.transpose(filter), 'same', 'symm')  # take y derivative
    return gx, gy


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
#    u_flat = u.flatten()
#    res = least_squares(loss, u_flat)
    np.save("result.npy", res.x)
    np.save("jacobien.npy", res.jac)
    return res.x

def basic_sharpening(img):
    img_blurred = filters.gaussian(img, sigma=1)
    gx = conv2(img_blurred, np.array([[-1, 0, 1]]), 'same')  # take x derivative
    gy = conv2(img_blurred, np.transpose(np.array([[-1, 0, 1]])), 'same')  # take y derivative
    cs = 2
    c1 = 0.5
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
    start = time.time()
    res = constrained_filter(d, g, w, u)
    end = time.time()
    print(str((end-start)/60) + "min")

    res = np.load("result.npy")
    res = np.reshape(res, (u.shape))
    print(np.sum(res - img))
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(img)
    axarr[1].imshow(res)
    plt.show()

def saliency_filter(img, s=None):
    if not s:
        sx,sy = salient_gradients(img)
    else:
        sx = s[0]
        sy = s[1]
    d = img
    gx, gy = get_gradients(img)
    gx = gx + c2 * sx
    gy = gy + c2 * sy
    wd = c1
    #TODO robust weighting scheme
    wx = np.ones(img.shape)
    wy = np.ones(img.shape)

def saliency_sharpening_noopt(img, s=None):
    try:
        num_channels = img.shape[2]
    except IndexError:
        num_channels = 1
        img = np.reshape(img, (img.shape[0], img.shape[1], 1))
    channels = []
    for i in range(num_channels):
        if not s:
            sx, sy = calc_saliency(img)
        else:
            sx = s[0]
            sy = s[1]
        out_img = img + 0.03 * (sx + sy)
        gx, gy = get_gradients(img)
        out_img_ns = img + 0.3 * (gx + gy)
    io.imshow(out_img)
    io.show()
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(out_img)
    axarr[1].imshow(out_img_ns)
    plt.show()
#    io.imsave("sharpened_test.jpg", np.clip(out_img, 0, 1))
#    io.imsave("sharpened_test_ns.jpg", np.clip(out_img_ns, 0, 1))


def npr_filter(img, sigma, e=None):
    """
    sigma is the abstraction amount
    c2 (>=1) controls the amount of exaggeration of local contrast across long edges
    c1 (>=0) controls how much the stylized image is allowed to drift from the input image
    """
    d = img
    gx_base, gy_base = get_gradients(img)

    exponent = np.divide(np.power(el, 2), np.power(-2 * sigma, 2 ))
    n = c2 * (1 - np.exp(exponent))

    gx = gx_base * np.power(np.cos(eo), 2) * n
    gy = gy_base * np.power(np.sin(eo), 2) * n

def npr_filter_noopt(img, ):
    pass


    


