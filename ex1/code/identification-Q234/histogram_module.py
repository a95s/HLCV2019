import numpy as np
import math
from numpy import histogram as hist

#  compute histogram of image intensities, histogram should be normalized so that sum of all values equals 1
#  assume that image intensity varies between 0 and 255
#
#  img_gray - input image in grayscale format
#  num_bins - number of bins in the histogram
def normalized_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'

    #bin_width = math.ceil(255.0 / num_bins)
    bin_width = 256.0 / num_bins
    hists = np.zeros(num_bins)
    bins = np.zeros(num_bins+1)
    img_flattened = img_gray.reshape((1, img_gray.shape[0]*img_gray.shape[1]))

    for px in img_flattened[0]:
        m = math.floor(px/bin_width)
        hists[m] = hists[m] + 1
        #print(px)
        #print(type(px))

    #print(img_gray.shape)
    #print(img_flattened.shape)
    # bins - X-coords
    # hists - height
    #print("will make {0} bins of {1} width".format(num_bins, bin_width))

    i = 0
    while i*bin_width <= 256:
        #print(i, (i*bin_width))
        bins[i] = (i*bin_width)
        i = i + 1

    # normalize
    hists = hists/np.sum(hists)
    #print(hists)
    #print(np.sum(hists))
    #print("printing lens/shape")
    #print(i)
    #print(len(bins))
    #print(len(hists))
    #print(bins.shape)
    #print(hists.shape)
    return hists, bins

#  compute joint histogram for each color channel in the image, histogram should be normalized so that sum of all values equals 1
#  assume that values in each channel vary between 0 and 255
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^3
def rgb_hist(img_color, num_bins):
    assert len(img_color.shape) == 3, 'image dimension mismatch'
    assert img_color.dtype == 'float', 'incorrect image type'

    # define a 3D histogram  with "num_bins^3" number of entries
    hists = np.zeros((num_bins, num_bins, num_bins))
    bin_width = 256.0 / num_bins
    # execute the loop for each pixel in the image
    for i in range(img_color.shape[0]):
        for j in range(img_color.shape[1]):
            # increment a histogram bin which corresponds to the value of pixel i,j; h(R,G,B)
            r = math.floor(img_color[i][j][0]/bin_width)
            g = math.floor(img_color[i][j][1]/bin_width)
            b = math.floor(img_color[i][j][2]/bin_width)

            #print(r,g,b)
            #print(img_color[i])
            #print(img_color[i][j])
            hists[r,g,b] += 1
            #exit(0)

    # normalize the histogram such that its integral (sum) is equal 1
    hists = hists/np.sum(hists)
    hists = hists.reshape(hists.size)
    return hists

#  compute joint histogram for r/g values
#  note that r/g values should be in the range [0, 1];
#  histogram should be normalized so that sum of all values equals 1
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2
def rg_hist(img_color, num_bins):

    assert len(img_color.shape) == 3, 'image dimension mismatch'
    assert img_color.dtype == 'float', 'incorrect image type'

    # define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))
    bin_width = 1.0 / num_bins
    # execute the loop for each pixel in the image
    for i in range(img_color.shape[0]):
        for j in range(img_color.shape[1]):
            # increment a histogram bin which corresponds to the value of pixel i,j; h(R,G,B)
            r = math.floor((img_color[i][j][0]/255.0)/bin_width)
            g = math.floor((img_color[i][j][1]/255.0)/bin_width)

            #print(r,g)
            #print(img_color[i])
            #print(img_color[i][j])
            hists[r,g] += 1
            #exit(0)
    # your code here
    hists = hists/np.sum(hists)
    hists = hists.reshape(hists.size)
    return hists


#  compute joint histogram of Gaussian partial derivatives of the image in x and y direction
#  for sigma = 7.0, the range of derivatives is approximately [-30, 30]
#  histogram should be normalized so that sum of all values equals 1
#
#  img_gray - input grayvalue image
#  num_bins - number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2
#
#  note: you can use the function gaussderiv.m from the filter exercise.
def dxdy_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'

    # compute the first derivatives
    from gauss_module import gaussderiv
    imgDx, imgDy = gaussderiv(img_gray, 7.0)

    # quantize derivatives to "num_bins" number of values
    bin_width = 256.0/num_bins

    # define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))

    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1]):
            # increment a histogram bin which corresponds to the value of pixel i,j; h(R,G,B)
            img_dx = math.floor(imgDx[i][j]/bin_width)
            img_dy = math.floor(imgDy[i][j]/bin_width)

            #print(r,g,b)
            #print(img_color[i])
            #print(img_color[i][j])
            hists[img_dx,img_dy] += 1
    # normalize
    hists = hists/np.sum(hists)
    #print(hists)
    hists = hists.reshape(hists.size)
    return hists

def is_grayvalue_hist(hist_name):
  if hist_name == 'grayvalue' or hist_name == 'dxdy':
    return True
  elif hist_name == 'rgb' or hist_name == 'rg':
    return False
  else:
    assert False, 'unknown histogram type'


def get_hist_by_name(img1_gray, num_bins_gray, dist_name):
  if dist_name == 'grayvalue':
    return normalized_hist(img1_gray, num_bins_gray)
  elif dist_name == 'rgb':
    return rgb_hist(img1_gray, num_bins_gray)
  elif dist_name == 'rg':
    return rg_hist(img1_gray, num_bins_gray)
  elif dist_name == 'dxdy':
    return dxdy_hist(img1_gray, num_bins_gray)
  else:
    assert 'unknown distance: %s'%dist_name
