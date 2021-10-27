# Computer Vision Assignment 02: Image Segmentation Report

## 2 Mean-Shift Algorithm (Total: 40 pts)

### 2.1 Implement the distance Function

* The distance function between two data points (row vectors) $\bf x_i$ and $\bf x_j$ is defined as:

$$
dist({\bf x_i}, {\bf x_j}) = ||({\bf x_i} - {\bf x_j})||^2_2
$$

* The broadcast property of pytorch tensor is applied to simplify the implementation.

### 2.2 Implement the gaussian Function

* The weights for point ${\bf x_j}$ given the center $\bf x_i$ are defined as ($\sigma$ is the bandwidth)
  $$
  w_j = {1 \over \sqrt{2\pi}\sigma^2} exp(-{ dist({\bf x_i}, {\bf x_j}) \over 2\sigma^2})
  $$

### 2.3 Implement the update point Function

* The update rule of $\bf x_i$ Is:
  $$
  {\bf x_i} = {\sum^n_{j=0} w_j {\bf x_j} \over \sum^n_{j=0} w_j} = { {1 \over \sum^n_{j=0} w_j}\bf{w_{x_i}}}X
  $$

### 2.4 Accelerating the Naive Implementation

* Running time of slow approach (CPU)

```shell
>>> python mean-shift.py
Elapsed time for mean-shift: 13.373154878616333
```

* Running time of fast approach (CPU) => obersevation: As the batch size increasing, the speed of the algorithm first increases and then decreases.

```shell
(batch_size = 100)
>>> python mean-shift.py
Elapsed time for mean-shift: 13.373154878616333
(batch_size = 500)
>>> python mean-shift.py
Elapsed time for mean-shift: 7.699587821960449
(batch_size = 1000)
>>> python mean-shift.py
lapsed time for mean-shift: 8.911517858505249
```



