import numpy as np
#
# compute chi2 distance between x and y
#
def dist_chi2(x,y):

    chi2_dist = 0.0

    for i in range(len(x)):
        if x[i] + y[i] != 0:
            chi2_dist += (x[i] - y[i])**2 / (x[i] + y[i])

    return chi2_dist
#
# compute l2 distance between x and y
#
def dist_l2(x,y):
  return np.sqrt(np.sum((x - y)**2))


#
# compute intersection distance between x and y
# return 1 - intersection, so that smaller values also correspond to more similart histograms
#
def dist_intersect(x,y):
    # link referred: https://mpatacchiola.github.io/blog/2016/11/12/the-simplest-classifier-histogram-intersection.html
   bins = len(x) # in our case both x and y have same number of bins

   intersection = 0
   for i in range(bins):
       intersection += min(x[i], y[i])
   return 1 - intersection

def get_dist_by_name(x, y, dist_name):
  if dist_name == 'chi2':
    return dist_chi2(x,y)
  elif dist_name == 'intersect':
    return dist_intersect(x,y)
  elif dist_name == 'l2':
    return dist_l2(x,y)
  else:
    assert 'unknown distance: %s'%dist_name
