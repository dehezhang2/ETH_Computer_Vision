import numpy as np

def BuildProjectionConstraintMatrix(points2D, points3D):

  # TODO
  # For each correspondence, build the two rows of the constraint matrix and stack them

  num_corrs = points2D.shape[0]
  constraint_matrix = np.zeros((num_corrs * 2, 12))
  
  for i in range(num_corrs):
    # TODO Add your code here 
    XT = np.hstack((points3D[i], 1))[None, :]
    x1, x2 = points2D[i][0], points2D[i][1]
    constraint_matrix[i*2] = (np.array([[0], [-1], [x2]]) @ XT).reshape(1, -1)
    constraint_matrix[i*2+1] = (np.array([[1], [0], [-x1]]) @ XT).reshape(1, -1)

  return constraint_matrix