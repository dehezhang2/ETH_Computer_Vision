# Computer Vision Assignment 04: Model Fitting & Multi-View Stereo

## 2 Model Fitting

### 2.1.3 Results

```shell
>>> python line_fitting.py
(500,)
Estimated coefficients (true, linear regression, RANSAC):
(ground truth) 									1 10 
(estimation from least-squares) 0.6159656578755457 8.96172714144364 (estimation from RANSAC)			  1.0103652266590768 9.807457331484198

```

## 3 Multi-View Stereo

### 3.2.2 Differentiable Warping

Equation of correspondence: if we assume the word coordinate is $\bf X$
$$
d_j{\bf p}= {\bf K}_0({\bf R}_0{\bf X}+{\bf t}_0) \mapsto d_j{\bf K}_0^{-1}{\bf p}= ({\bf R}_0{\bf X}+{\bf t}_0)\\
{\bf p}_{ij}= {\bf K}_i({\bf R}_{0,i}({\bf R}_0{\bf X}+{\bf t}_0) + {\bf t}_{0,i}) = {\bf K}_i(d_j{\bf R}_{0,i}{\bf K}_0^{-1}{\bf p} + {\bf t}_{0,i})\\
$$
