import time
import os
import random
import math
import torch
import numpy as np

# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale

def distance(x, X):
    dist = (X-x).pow(2).sum(axis=1).sqrt()
    return dist

def distance_batch(x, X):
    X_r = X.reshape(1, -1, X.shape[1])
    x_r = x.reshape(-1, 1, x.shape[1])
    dist = (X_r-x_r).pow(2).sum(axis=-1).sqrt()
    return dist

def gaussian(dist, bandwidth):
    weights = torch.exp( -0.5 * (dist/bandwidth).pow(2) ) / (bandwidth * math.sqrt(2*math.pi))
    return weights

def update_point(weight, X):
    X_i_updated = torch.matmul(weight,X) / weight.sum()
    return X_i_updated

def update_point_batch(weight, X):
    X_i_updated = torch.matmul(weight,X)
    res = torch.transpose(X_i_updated, 0, 1) / weight.sum(axis=1)
    return torch.transpose(res, 0, 1)


def meanshift_step(X, bandwidth=2.5):
    X_ = X.clone()
    for i, x in enumerate(X):
        dist = distance(x, X)
        weight = gaussian(dist, bandwidth)
        X_[i] = update_point(weight, X)
    return X_

def meanshift_step_batch(X, bandwidth=2.5):
    sample_per_batch = 200
    total = X.shape[0]
    X_ = X.clone()
    for i in range(0, total, sample_per_batch):
        interval = slice(i, min(i+sample_per_batch, total))
        dist = distance_batch(X[interval],X)
        weight = gaussian(dist, bandwidth)
        X_[interval] = update_point_batch(weight, X)
    return X_

def meanshift(X):
    X = X.clone()
    for _ in range(20):
        # X = meanshift_step(X)   # slow implementation
        X = meanshift_step_batch(X)   # fast implementation
    return X

scale = 0.25    # downscale the image to run faster

# Load image and convert it to CIELAB space
image = rescale(io.imread('cow.jpg'), scale, multichannel=True)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image

# Run your mean-shift algorithm
t = time.time()
X = meanshift(torch.from_numpy(image_lab)).detach().cpu().numpy()
# X = meanshift(torch.from_numpy(data).cuda()).detach().cpu().numpy()  # you can use GPU if you have one
t = time.time() - t
print ('Elapsed time for mean-shift: {}'.format(t))

# Load label colors and draw labels as an image
colors = np.load('colors.npz')['colors']
colors[colors > 1.0] = 1
colors[colors < 0.0] = 0

centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)

result_image = colors[labels].reshape(shape)
result_image = rescale(result_image, 1 / scale, order=0, multichannel=True)     # resize result image to original resolution
result_image = (result_image * 255).astype(np.uint8)
io.imsave('result.png', result_image)
