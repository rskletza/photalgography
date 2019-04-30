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
import scipy.sparse as sparse
import scipy.linalg as linalg

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
    gx, gy = get_gradients(img)

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
    m0_qs = get_qs(0)
    m1_qs = get_qs(1)
    w_alphas_0 = w_alpha_vectorized(m0_qs)
    w_alphas_1 = w_alpha_vectorized(m1_qs)
    w_thetas_0 = w_theta_vectorized(m0_qs)
    w_thetas_1 = w_theta_vectorized(m1_qs)
    interpol_magn_0 = interpolate_qs(m0_qs, normalized_magn)
    interpol_magn_1 = interpolate_qs(m1_qs, normalized_magn)
    for i in range(40):
        m0_new = w_alphas_0 * w_thetas_0 * (interpol_magn_0 + interpolate_qs(m0_qs, m0_prev))
        m1_new = w_alphas_1 * w_thetas_1 * (interpol_magn_1 + interpolate_qs(m1_qs, m1_prev))
        m0_prev = m0_new
        m1_prev = m1_new

    end = time.time()
    print("execution time: " + str(end - start) + "s")

    lengths = m0_new + m1_new + normalized_magn
    np.save("lengths-out.npy", lengths)
#    lengths = np.load("./lengths_pixi_grayscale.npy")

    sx = np.power(np.cos(orient), 2) * lengths * gx
    sy = np.power(np.sin(orient), 2) * lengths * gy

#    f, axarr = plt.subplots(1,3)
##    axarr[0].imshow(img)
#    axarr[0].imshow(gx + gy + 0.5)
##    axarr[2].imshow(normalized_magn)
#    axarr[1].imshow(lengths)
#    axarr[2].imshow(sx + sy)
#    plt.show()

#    io.imsave("0_grayscale.jpg", img)
#    io.imsave("1_gradients.jpg", exposure.rescale_intensity(gx + gy + 0.5))
#    io.imsave("2_normalized_gradients.jpg", exposure.rescale_intensity(normalized_magn))
#    io.imsave("3_lengths.jpg", exposure.rescale_intensity(lengths))
#    io.imsave("4_final.jpg", exposure.rescale_intensity(np.abs(sx + sy)))

    el = lengths
    eo = orient

    return sx, sy, el, eo

def get_gradients(img):
    """
    calculate gradients of the image. The method used by this function needs to match the equations of the matrix of equations (in this case, subtracting the right/lower pixel from the left/upper pixel to get the gradient of a pixel)
    """
#    gx = np.roll(img, -1, axis=0) - np.roll(img, 1, axis=0)
#    gy = np.roll(img, -1, axis=1) - np.roll(img, 1, axis=1)
#
#    #correct the edges to 0
#    gx[0,:] = 0
#    gx[gx.shape[0]-1,:] = 0
#
#    gy[:,0] = 0
#    gy[:,gy.shape[1]-1] = 0

    gx = np.diff(img, 1, 1);
    gy = np.diff(img, 1, 0);

    gx = np.hstack((gx, np.zeros((img.shape[0], 1))))
    gy = np.vstack((gy, np.zeros((1, img.shape[1]))))

#    f, axarr = plt.subplots(1,4)
#    axarr[0].imshow(gx + 0.5)
#    axarr[1].imshow(gy + 0.5)
#    axarr[2].imshow(gx2 + 0.5)
#    axarr[3].imshow(gy2 + 0.5)


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

def build_Ad(wd, d):
    """
    creates the part of the equation (Ax = b) responsible for the pixel values
    """
    bd = d.flatten()
    Ad = sparse.diags(wd.flatten())
    return Ad, bd

def build_Agx(wgx, gx):
    """
    builds the equations for the system Ax = b responsible for the x derivative
    omits the outer edge of the image, because the derivative is not well defined there
    """
    n = gx.shape[0] * gx.shape[1]
    bgx = gx[:,1:gx.shape[1]-1].flatten()
    Agx = sparse.dok_matrix((bgx.shape[0], n))
    A_row = 0
#    for y in range(gx.shape[0]):
#        for x in range(1,gx.shape[1]-1):
#            Agx[A_row, flatten_index(x+1,y,gx.shape)] = wgx[y,x]
#            Agx[A_row, flatten_index(x-1,y,gx.shape)] = -(wgx[y,x])
#            A_row += 1
    for y in range(gx.shape[0]):
        for x in range(1,gx.shape[1]-1):
            Agx[A_row, flatten_index(x+1,y,gx.shape)] = wgx[y,x]
            Agx[A_row, flatten_index(x,y,gx.shape)] = -(wgx[y,x])
            A_row += 1

    return Agx, bgx

def build_Agy(wgy, gy):
    """
    builds the equations for the system Ax = b responsible for the y derivative
    omits the outer edge of the image, because the derivative is not well defined there
    """
    n = gy.shape[0] * gy.shape[1]
    bgy = gy[1:gy.shape[0]-1,:].flatten()
    Agy = sparse.dok_matrix((bgy.shape[0], n))
    A_row = 0
    for y in range(1,gy.shape[0]-1):
        for x in range(gy.shape[1]):
            Agy[A_row, flatten_index(x,y+1,gy.shape)] = wgy[y,x]
            Agy[A_row, flatten_index(x,y,gy.shape)] = -(wgy[y,x])
            A_row += 1

    return Agy, bgy

def flatten_index(x,y,shape):
    """
    given a 2D index, returns the index of that element in the flattened array
    shape passed as (y_max, x_max), like numpy.shape
    """
    return y*shape[1] + x

def solve_for_constraints(d, g, w, img):
    """
    d: intensity constraints (1 channel)
    g: gradient constraints (h and v, 2 channels)
    w: weights of d and g (3 channels)
    img: input image (used to adjust the skewed output range)
    """
    gx = g[:,:,0]
    gy = g[:,:,1]
    wd = w[:,:,0]
    wgx = w[:,:,1]
    wgy = w[:,:,2]

    Ad, bd = build_Ad(wd, d)
    Agx, bgx = build_Agx(wgx, gx)
    Agy, bgy = build_Agy(wgy, gy)

    A = sparse.vstack((Ad, Agx, Agy))
    b = np.concatenate((bd, bgx, bgy))

    res = sparse.linalg.lsqr(A, b, atol=0, btol=0, conlim=0)[0]
    res = np.reshape(res, img.shape)

    #correct image range
    input_mean = np.mean(img[:]);
    res_mean = np.mean(res[:]);

    offset = input_mean - res_mean;
    fixed_res = res + offset
    fixed_res = np.maximum(0, np.minimum(1, fixed_res))

    return fixed_res

def basic_sharpening(img):
    gx, gy = get_gradients(img)
    cg = 1.5
    wg = 1
    cd = 0.6#3e-02
    d = img
    g = np.dstack((gx * cg, gy * cg))
    
    w = np.dstack((np.full(img.shape, cd), np.full(img.shape, wg), np.full(img.shape, wg)))
    start = time.time()
    res = solve_for_constraints(d, g, w, img)
    end = time.time()
    print(str(end-start) + "s")
#    io.imshow(res)
#    io.show()

    return res

def salient_sharpening(img, params=None):
    gx, gy = get_gradients(img)
    if params is None:
        sx,sy,_,_ = calc_saliency(img)
    else:
        sx = params[0]
        sy = params[1]
    cg = 0.4
    wg = 1
    cd = 0.8#3e-02
    d = img
    g = np.dstack((gx + cg * sx, gy + cg * sy))
    
    w = np.dstack((np.full(img.shape, cd), np.full(img.shape, wg), np.full(img.shape, wg)))
    start = time.time()
    res = solve_for_constraints(d, g, w, img)
    end = time.time()
    print(str(end-start) + "s")
#    io.imshow(res)
#    io.show()

    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(img)
    axarr[1].imshow(res)
    plt.show()
    if params is not None:
        return res
    else:
        return res, np.array([sx, sy])

def saliency_sharpening_noopt(img, s=None):
    try:
        num_channels = img.shape[2]
    except IndexError:
        num_channels = 1
        img = np.reshape(img, (img.shape[0], img.shape[1], 1))

    out_channels = []
    out_channels_ns = []
    for i in range(num_channels):
        channel = img[:,:,i]
        if not s:
            sx, sy, el, eo = calc_saliency(channel)
        else:
            sx = s[0]
            sy = s[1]
        out_channels.append(channel + 0.03 * (sx + sy))
        gx, gy = get_gradients(channel)
        out_channels_ns.append(channel + 0.3 * (gx + gy))

    if num_channels > 1:
        out_img = np.dstack(out_channels)
        out_img_ns = np.dstack(out_channels_ns)
    else:
        out_img = out_channels[0]
        out_img_ns = out_channels[0]

    io.imshow(out_img)
    io.show()
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(out_img)
    axarr[1].imshow(out_img_ns)
    plt.show()
    io.imsave("./sharpening/out.jpg", np.clip(out_img, 0, 1))
    io.imsave("./sharpening/out_ns.jpg", np.clip(out_img_ns, 0, 1))


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

def filter_img(img, filter, params=None):
    try:
        num_channels = img.shape[2]
    except IndexError:
        num_channels = 1
        img = np.reshape(img, (img.shape[0], img.shape[1], 1))

    out_channels = []
    out_channels_ns = []
    for i in range(num_channels):
        channel = img[:,:,i]
        if params is not None:
            filtered = filter(img[:,:,i], params[i])
        else:
            filtered, new_params = filter(img[:,:,i])
            np.save("./params/params" + str(i) + ".npy", new_params)
        out_channels.append(filtered)

    if num_channels > 1:
        out_rgb = np.dstack(out_channels)
    else:
        out_rgb = out_channels[0]

    return out_rgb
    


