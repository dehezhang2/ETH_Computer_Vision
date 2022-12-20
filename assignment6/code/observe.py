import numpy as np
from chi2_cost import chi2_cost
from color_histogram import color_histogram

def observe(particles, frame, bbox_height, bbox_width, hist_bin, target_hist, sigma_observe):
    particles_w = np.zeros((particles.shape[0], 1))

    xmins = np.maximum(particles[:, 0] - bbox_width/2, 0)
    xmaxs = np.minimum(particles[:, 0] + bbox_width/2, frame.shape[1]-1)
    ymins = np.maximum(particles[:, 1] - bbox_height/2, 0)
    ymaxs = np.minimum(particles[:, 1] + bbox_height/2, frame.shape[0]-1)

    for i in range(particles_w.shape[0]):
        sample_hist = color_histogram(int(xmins[i]), int(ymins[i]), int(xmaxs[i]), int(ymaxs[i]), frame, hist_bin)
        dist = chi2_cost(target_hist, sample_hist)
        particles_w[i] = 1/(np.sqrt(2*np.pi)*sigma_observe) * np.exp(-dist**2/( 2*(sigma_observe**2) ))
    
    particles_w = particles_w/np.sum(particles_w)
    return particles_w
