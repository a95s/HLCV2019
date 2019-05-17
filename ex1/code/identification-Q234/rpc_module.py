import numpy as np
import matplotlib.pyplot as plt

import histogram_module
import dist_module
import match_module

#
# compute and plot the recall/precision curve
#
# D - square matrix, D(i, j) = distance between model image i, and query image j
#
# note: assume that query and model images are in the same order, i.e. correct answer for i-th query image is the i-th model image
#
def plot_rpc(D, plot_color):
  recall = []
  precision = []
  total_imgs = D.shape[1]

  num_images = D.shape[0]
  assert(D.shape[0] == D.shape[1])

  labels = np.diag([1]*num_images)

  # flattened distance matrix and labels
  #d = D.reshape(D.size)
  #l = labels.reshape(labels.size)

  # sort indices in ascending order to sort the horizontal axis.
  #sortidx = d.argsort()
  #d = d[sortidx]
  #l = l[sortidx]

  tp = 0
  """for idx in range(len(d)):
    tp = tp + l[idx]

    # compute precision and recall values and append them to "recall" and "precision" vectors
    precision = tp / (tp + fp)
    recall    = tp / (tp + fn)
  """
  p = 0
  r = 0

  # to get range of distances in D
  max_dist = np.max(D)
  for threshold in np.linspace(0, max_dist, num=40):

    TP = np.sum((np.diag(D) < threshold).astype(float))
    FN = np.sum((np.diag(D) > threshold).astype(float))

    diagonal_mask = np.ones((num_images,num_images)) - np.diag([1]*num_images)
    FP = np.sum((D < threshold).astype(float) * diagonal_mask ) / num_images
    #print(TP,FP,FN)
    if(TP + FP) == 0 or (TP + FN) == 0:
        continue

    p = TP / (TP + FP)
    r = TP / (TP + FN)

    precision.append(p)
    recall.append(r)

  plt.plot([1-precision[i] for i in range(len(precision))], recall, plot_color+'-')


def compare_dist_rpc(model_images, query_images, dist_types, hist_type, num_bins, plot_colors):

  assert len(plot_colors) == len(dist_types)

  for idx in range( len(dist_types) ):

    [best_match, D] = match_module.find_best_match(model_images, query_images, dist_types[idx], hist_type, num_bins)

    plot_rpc(D, plot_colors[idx])


  plt.axis([0, 1, 0, 1]);
  plt.xlabel('1 - precision');
  plt.ylabel('recall');

  plt.legend( dist_types, loc='best')
