# wd-weno

WENO based adaptive image zooming algorithm

<img width="300px" src="images/fig_02_head.png">
<img width="300px" src="images/fig_02_head_q16x_exponent_2.0.png">



## Overview
wd-weno is an open-source Python module for polynomial interpolation algorithm based on the WENO algorithm for upsampling images.

## Features
- Low complexity and memory load nonlinear 2D interpolation method
- Non-oscillatory high order approximation even in the presence of strong jump discontinuities
- Reduces numerical artifacts like staircase effects by adapting the direction of interpolation to the data
- Straightforward parallelization and vectorization of the algorithm

## Getting started

### Prerequisites
- Python 3.12 or later
- Dependencies
  - NumPy
  - SciKit-image
  - Numba

### Installation
Download `wdweno` directory and start using it.

### Quick Start
```
import wdweno

wdweno.wdweno(in_image=<path to input image>, out_image=<path to output image>,
                method='2x', scale_exp=1, beta=2)
```
Or simply take a look at the examples directory.


## License
wd-weno is licensed under the [GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/#).
You are free to use, modify, and distribute this software under the terms of the GPLv3 license. P
lease note that derivative works must also be licensed under the GPL.

## Citing wdweno
```
@software{wdweno,
  author       = {Bojan Crnković and Jerko Škifić and Tina Bosner},
  title        = {wdweno: WENO Based Adaptive Image Zooming Algorithm},
  year         = {2024},
  version      = {0.1.0},
  doi          = {10.1016/j.amc.2024.129228},
  url          = {https://github.com/jskific/wdweno}
}
```



## Authors and acknowledgment
This project was developed by
- Bojan Crnković
- Jerko Škifić
- Tina Bosner
