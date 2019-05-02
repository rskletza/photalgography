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

def calc_gradients(img):
    """
    calculate gradients of the image. The method used by this function needs to match the equations of the matrix of equations (in this case, subtracting the right/lower pixel from the left/upper pixel to get the gradient of a pixel)
    """
    gx = np.diff(img, 1, 1);
    gy = np.diff(img, 1, 0);

    #correct the image back to regular size
    gx = np.hstack((gx, np.zeros((img.shape[0], 1))))
    gy = np.vstack((gy, np.zeros((1, img.shape[1]))))

    return gx, gy

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

def basic_sharpening(img):
    gx, gy = calc_gradients(img)
    cg = 1.5
    wg = 1
    cd = 0.7
    d = img
    g = np.dstack((gx * cg, gy * cg))
    
    w = np.dstack((np.full(img.shape, cd), np.full(img.shape, wg), np.full(img.shape, wg)))
    res = solve_for_constraints(d, g, w, img)
    return res, None

def salient_sharpening(img, params=None):
    gx, gy = calc_gradients(img)
    if params is None:
        el, eo = calc_edge_params(gx, gy)
        sx,sy = calc_saliency(img, [el, eo])
    else:
        el = params[0]
        eo = params[1]
        sx,sy = calc_saliency(img, [el, eo])
    cg = 0.6
    wg = 1
    cd = 0.8
    d = img
    g = np.dstack((gx + cg * sx, gy + cg * sy))
    
    w = np.dstack((np.full(img.shape, cd), np.full(img.shape, wg), np.full(img.shape, wg)))
    res = solve_for_constraints(d, g, w, img)

    if params is not None:
        return res
    else:
        return res, np.array([el, eo])

def to_pencil(edges):
    pass

def calc_robust_weights(cd, gx, gy, gx_constraint, gy_constraint, b):
    x_denominator = np.power(np.abs(gx - gx_constraint) + 1, b)
    wx = np.divide(1, x_denominator)

    y_denominator = np.power(np.abs(gy - gy_constraint) + 1, b)
    wy = np.divide(1, y_denominator)

    return np.dstack((np.full(gx.shape, cd), wx, wy))

def npr_filter(img, params=None):
    """
    sigma is the abstraction amount
    cg (>=1) controls the amount of exaggeration of local contrast across long edges
    cd (>=0) controls how much the stylized image is allowed to drift from the input image
    """
    gx, gy = calc_gradients(img)
    if params is None:
        print("calculating edge parameters")
        el, eo = calc_edge_params(gx, gy)
    else:
        el = params[0]
        eo = params[1]
    cd = 0.25
    d = img

    cg = 1.9
    wg = 1

    sigma = 10
    exponent = np.divide(np.power(el, 2), -2 * np.power(sigma, 2 ))
    n = cg * (1 - np.exp(exponent))
    gx_constraint = gx * np.power(np.cos(eo), 2) * n
    gy_constraint = gy * np.power(np.sin(eo), 2) * n
    edges = gx_constraint + gy_constraint
    edges = edges + np.roll(edges, 1, axis=0) + np.roll(edges, -1, axis=0) + np.roll(edges, 1, axis=1) + np.roll(edges, -1, axis=1)
    edges = np.abs(1 - 20 * edges)
#    print(edges)
#    print(np.min(edges), np.max(edges))
    io.imshow(edges)
    io.show()

    g = np.dstack((gx_constraint, gy_constraint))
    
    w = calc_robust_weights(cd, gx, gy, gx_constraint, gy_constraint, 9)
    res = solve_for_constraints(d, g, w, img)
    res = res * edges

    if params is not None:
        return res
    else:
        return res, np.array([el, eo])

def filter_img(img, filter, params=None):
    try:
        num_channels = img.shape[2]
    except IndexError:
        num_channels = 1
        img = np.reshape(img, (img.shape[0], img.shape[1], 1))

    out_channels = []
    out_channels_ns = []
    for i in range(num_channels):
        print(i)
        channel = img[:,:,i]
        if params is not None:
            filtered = filter(img[:,:,i], params[i])
        else:
            filtered, new_params = filter(img[:,:,i])
            #if filter calculated new parameters, save them for later
            if new_params is not None: 
                np.save("./params/params_test" + str(i) + ".npy", new_params)
        out_channels.append(filtered)

    if num_channels > 1:
        out_rgb = np.dstack(out_channels)
    else:
        out_rgb = out_channels[0]

    return out_rgb
    
def calc_edge_params(gx, gy):
    """
    estimates the length of continuous edges and returns two arrays: e_l (length of edge containing certain pixel) and e_o(orientation of edge at this pixel)
    """
    def get_qs(direction):
        """
        get the exact index of q, calculated by moving sqrt(2) along the edge in a specified direction (either in the direction of the edge (orientation + pi/2), or in the other direction of the edge (orientation + p/2 + pi))
        """
        edge_dirs = orient + np.pi/2.0 + direction*np.pi
        positions = np.indices((gx.shape))
        q_vector_x = np.cos(edge_dirs) * np.sqrt(2)
        q_vector_y = np.sin(edge_dirs) * np.sqrt(2)
        q_vectors = np.dstack((q_vector_y, q_vector_x))
        #creates an array with the position (index) at each index
        indices = np.swapaxes(np.swapaxes(np.indices((orient.shape)), 0, 1), 1, 2)
        qs_flippedindex = indices + q_vectors
        qs_indices = np.flip(qs_flippedindex, axis = 2)
        return qs_indices

    def interpolate_qs(qs, array):
        interpol_array = np.zeros((gx.shape))
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

    def w_theta_vectorized(qs):
        """
        measures similarity of the local edge orientations (using gradient orientation, but that should not make a difference)
        """
        p_theta = orient
        q_theta = interpolate_qs(qs, orient)
        weight = np.exp( -1 * np.power((p_theta -  q_theta), 2) / (2 * np.pi/5.0))
        return weight

    def w_alpha_vectorized(qs):
        #TODO don't know what this is supposed to do
        return np.ones(gx.shape)
    
    magn, orient = cv2.cartToPolar(gx, gy, angleInDegrees=False)

    normalized_magn = block_normalize(magn)

    m0_prev = np.zeros((gx.shape))
    m1_prev = np.zeros((gx.shape))
    m0_new = np.zeros((gx.shape))
    m1_new = np.zeros((gx.shape))
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

    lengths = m0_new + m1_new + normalized_magn
#    np.save("lengths-out.npy", lengths)
#    lengths = np.load("./lengths_pixi_grayscale.npy")

    return lengths, orient

def calc_saliency(img, e=None):
    """
    calculate the saliency of the local gradients: long, coherent edges are perceptually more important to human perception. 
    """
    gx, gy = calc_gradients(img)
    if e is None:
        el, eo = calc_edge_params(gx, gy)
    else:
        el = e[0]
        eo = e[1]
    sx = np.power(np.cos(eo), 2) * el * gx
    sy = np.power(np.sin(eo), 2) * el * gy
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
