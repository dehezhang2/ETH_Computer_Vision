import numpy as np
def resample(particles, particles_w):
    sample_size = particles.shape[0]
    indices = np.random.choice(np.linspace(0, sample_size-1, sample_size, dtype=int), size=sample_size, p=particles_w[:, 0])

    sampled_particles = particles[indices]
    sampled_particles_w = particles_w[indices]
    sampled_particles_w = sampled_particles_w/np.sum(sampled_particles_w)

    return sampled_particles, sampled_particles_w