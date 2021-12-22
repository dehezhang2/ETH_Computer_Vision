import numpy as np
def estimate(particles, particles_w):
    mean_particle = (particles.T @ particles_w).T
    return mean_particle