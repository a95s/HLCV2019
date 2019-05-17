import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import histogram_module
import dist_module

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

#
# model_images - list of file names of model images
# query_images - list of file names of query images
#
# dist_type - string which specifies distance type:  'chi2', 'l2', 'intersect'
# hist_type - string which specifies histogram type:  'grayvalue', 'dxdy', 'rgb', 'rg'
#
# note: use functions 'get_dist_by_name', 'get_hist_by_name' and 'is_grayvalue_hist' to obtain
#       handles to distance and histogram functions, and to find out whether histogram function
#       expects grayvalue or color image
#

def find_best_match(model_images, query_images, dist_type, hist_type, num_bins):

  hist_isgray = histogram_module.is_grayvalue_hist(hist_type)


  model_hists = compute_histograms(model_images, hist_type, hist_isgray, num_bins)
  query_hists = compute_histograms(query_images, hist_type, hist_isgray, num_bins)

  D = np.zeros((len(model_images), len(query_images)))

  for m in range(len(model_images)):
      for q in range(len(query_images)):
          D[m,q] = dist_module.get_dist_by_name(model_hists[m],query_hists[q],dist_type)

  best_match = np.argmin(D, axis=0)
  return best_match, D

def compute_histograms(image_list, hist_type, hist_isgray, num_bins):

  image_hist = []

  # compute hisgoram for each image and add it at the bottom of image_hist
  for img_path in image_list:
      img = np.array(Image.open(img_path)).astype('double')
      if hist_isgray:
          img = rgb2gray(img)

          if hist_type == 'grayvalue':
              # discard bin info returned by gray value hist function
              hist, _ = histogram_module.get_hist_by_name(img, num_bins, hist_type)
              image_hist.append(hist)
          else: #dxdy
              image_hist.append(histogram_module.get_hist_by_name(img, num_bins, hist_type))
      else:
          image_hist.append(histogram_module.get_hist_by_name(img, num_bins, hist_type))

  return image_hist

#
# for each image file from 'query_images' find and visualize the 5 nearest images from 'model_image'.
#
# note: use the previously implemented function 'find_best_match'
# note: use subplot command to show all the images in the same Python figure, one row per query image
#

def show_neighbors(model_images, query_images, dist_type, hist_type, num_bins):

  plt.figure()

  num_nearest = 5  # show the top-5 neighbors
  best_match, D = find_best_match(model_images, query_images, dist_type, hist_type, num_bins)
  top_5_neighbours_idx = np.argsort(D,axis=0)[:num_nearest]

  r = 0
  for q in query_images:
      neighbors_of_q = top_5_neighbours_idx[:,r]
      plt.subplot(len(query_images),num_nearest+1, (num_nearest+1)*r + 1)
      plt.imshow(np.array(Image.open(q)), vmin=0, vmax=255)
      i = 6*r + 1
      for nq in neighbors_of_q:
          plt.subplot(len(query_images),6, i+1)
          plt.imshow(np.array(Image.open(model_images[nq])), vmin=0, vmax=255)
          i = i + 1
      r = r + 1
  plt.show()

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
