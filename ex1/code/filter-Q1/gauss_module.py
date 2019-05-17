# import packages: numpy, math (you might need pi for gaussian functions)
import numpy as np
import math
from scipy.signal import convolve2d as conv2

def gauss(sigma):

    x = np.arange(-3.0 * sigma, 3.0 * sigma, 1)
    G = np.exp(-(x**2 / (2.0 * sigma**2))) / (np.sqrt(2*np.pi) * sigma)

    return G, x

def gaussderiv(img, sigma):
    G, _ = gauss(sigma)
    D, _ = gaussdx(sigma)
    G = G.reshape(1, G.size)
    D = D.reshape(1, D.size)

    imgDx = conv2(conv2(img, G, 'same'), D.T, 'same')
    imgDy = conv2(conv2(img, G.T, 'same'), D, 'same')
    return imgDx, imgDy

def gaussdx(sigma):

    x = np.arange(-3.0 * sigma, 3.0 * sigma, 1)
    D = x * np.exp(-(x**2 / (2.0 * sigma**2))) / (-np.sqrt(2*np.pi) * sigma**3)
    return D, x

def gaussianfilter(img, sigma):

    G, x = gauss(sigma)
    Ga = G.reshape(G.shape[0],1)
    Gb = G.reshape(1,G.shape[0])
    # because of gaussian separability we can apply 1-D kernel twice
    # to get 2D gaussian blur output
    outimage = conv2(conv2(img, Ga, 'same'), Gb, 'same')
    return outimage
