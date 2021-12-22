import numpy as np
def propagate(particles, frame_height, frame_width, params):
    A = np.identity(particles.shape[1])
    w = np.zeros(particles.shape)
    if params['model'] == 1:
        A[0, 2] = 1
        A[1, 3] = 1
        w[:, 2:4] += np.random.normal(0, params['sigma_velocity'], w[:, 2:4].shape)
    w[:, 0:2] += np.random.normal(0, params['sigma_position'], w[:, 0:2].shape)
    next_particles = (A@particles.T).T + w
    next_particles[:, 0] = np.minimum(np.maximum(next_particles[:, 0], 0), frame_width-1)
    next_particles[:, 1] = np.minimum(np.maximum(next_particles[:, 1], 0), frame_height-1)

    return next_particles
