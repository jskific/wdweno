from wdweno.coo_dx_utils import create_coo_slanted, create_coo_ortho, create_dx
from numba import jit, void, int64, float64
import numpy as np
from time import perf_counter

from wdweno.init_img_arrays import (init_nscale_arrays,
                             create_initial_scaled_image,
                             pad_edge_image,
                             fill_lin_interp_padding)


@jit(
    void(int64[:, :, :], float64[:, :], float64,
         int64, int64, int64, int64,
         float64[:, :], float64[:, :], float64[:, :], float64[:]),
    nopython=True)
def slanted_nx(coo_kose, dx_arr, exponent,
               n2x, n2y, stride, pad_offset,
               image2x, SM, SMTotal, y):
    offset = pad_offset + int(stride / 2)  # + 2*(scale_ex-1)
    for coo_idx in range(len(coo_kose)):
        p = coo_kose[coo_idx]
        dx = dx_arr[coo_idx]
        for i in range(offset, n2x - offset, stride):
            for j in range(offset, n2y - offset, stride):
                y[0] = image2x[p[1, 0] + i, p[1, 1] + j] - image2x[p[0, 0] + i, p[0, 1] + j]
                y[1] = image2x[p[2, 0] + i, p[2, 1] + j] - image2x[p[1, 0] + i, p[1, 1] + j]
                y[1] -= y[0]
                SM[i][j] = (y[1] ** 2) / 3. + y[0] ** 2

        for i in range(offset, n2x - offset, stride):
            for j in range(offset, n2y - offset, stride):
                y[0] = image2x[p[0][0] + i, p[0][1] + j]
                y[1] = image2x[p[1][0] + i, p[1][1] + j]
                y[2] = image2x[p[2][0] + i, p[2][1] + j]
                # quadratic interp
                v = (3. * y[0] + 6. * y[1] - y[2]) / 8.
                si = SM[i][j]
                # calc smoothness indicator
                si += (SM[i + stride, j] +
                       SM[i - stride, j] +
                       SM[i, j + stride] +
                       SM[i, j - stride]) / 8
                w = 1. / (1.e-12 + si) ** exponent

                SMTotal[i][j] += w
                image2x[i][j] += v * w

    image2x[offset: n2x - offset: stride,
    offset: n2y - offset: stride] /= SMTotal[offset: n2x - offset: stride, offset: n2y - offset: stride]


@jit(
    void(int64[:, :, :], float64[:, :], float64,
         int64, int64, int64, int64,
         float64[:, :], float64[:, :], float64[:, :], float64[:]),
    nopython=True)
def horz_vert_nx(coo_ravne, dx_arr,
                 exponent, n2x, n2y,
                 stride, pad_offset,
                 image2x, SM, SMTotal, y):
    offset = pad_offset

    irangev = range(offset, n2x - offset, stride)
    jrangev = range(offset + int(stride / 2), n2y - (offset + int(stride / 2)), stride)

    irangeh = range(offset + int(stride / 2), n2x - (offset + int(stride / 2)), stride)
    jrangeh = range(offset, n2y - offset, stride)

    sm_stride = int(stride / 2)
    for coo_idx in range(len(coo_ravne)):
        p = coo_ravne[coo_idx]
        dx = dx_arr[coo_idx]

        for i in irangev:
            for j in jrangev:
                y[0] = image2x[p[1, 0] + i, p[1, 1] + j] - image2x[p[0, 0] + i, p[0, 1] + j]
                y[1] = image2x[p[2, 0] + i, p[2, 1] + j] - image2x[p[1, 0] + i, p[1, 1] + j]
                y[1] -= y[0]
                SM[i][j] = (y[1] ** 2) / 3. + y[0] ** 2
        for i in irangeh:
            for j in jrangeh:
                y[0] = image2x[p[1, 0] + i, p[1, 1] + j] - image2x[p[0, 0] + i, p[0, 1] + j]
                y[1] = image2x[p[2, 0] + i, p[2, 1] + j] - image2x[p[1, 0] + i, p[1, 1] + j]
                y[1] -= y[0]
                SM[i][j] = (y[1] ** 2) / 3. + y[0] ** 2

        for i in irangev:
            for j in jrangev:
                y[0] = image2x[p[0, 0] + i, p[0, 1] + j]
                y[1] = image2x[p[1, 0] + i, p[1, 1] + j]
                y[2] = image2x[p[2, 0] + i, p[2, 1] + j]
                # quadratic interp
                v = (3. * y[0] + 6. * y[1] - y[2]) / 8.
                si = SM[i][j]
                # calc smoothness indicator
                si += (SM[i + sm_stride, j + sm_stride] +
                       SM[i - sm_stride, j - sm_stride] +
                       SM[i + sm_stride][j - sm_stride] +
                       SM[i - sm_stride][j + sm_stride]) / 8
                w = 1. / (1.e-12 + si) ** exponent
                SMTotal[i][j] += w
                image2x[i][j] += v * w

        for i in irangeh:
            for j in jrangeh:
                y[0] = image2x[p[0, 0] + i, p[0, 1] + j]
                y[1] = image2x[p[1, 0] + i, p[1, 1] + j]
                y[2] = image2x[p[2, 0] + i, p[2, 1] + j]
                # quadratic interp
                v = (3. * y[0] + 6. * y[1] - y[2]) / 8.
                si = SM[i][j]
                # calc smoothness indicator
                si += (SM[i + sm_stride, j + sm_stride] +
                       SM[i - sm_stride, j - sm_stride] +
                       SM[i + sm_stride][j - sm_stride] +
                       SM[i - sm_stride][j + sm_stride]) / 8
                w = 1. / (1.e-12 + si) ** exponent
                SMTotal[i][j] += w
                image2x[i][j] += v * w

    for i in irangev:
        for j in jrangev:
            image2x[i][j] /= SMTotal[i][j]
    for i in irangeh:
        for j in jrangeh:
            image2x[i][j] /= SMTotal[i][j]


@jit(
    float64(float64, float64[:], float64),
    nopython=True)
def scale_weno1d(d, y, exponent):
    q1 = y[0] * d * (d - 1) / 2 - y[1] * (d - 1) * (1 + d) + y[2] * d * (1 + d) / 2
    q2 = y[1] * (d - 1) * (d - 2) / 2 - y[2] * d * (d - 2) + y[3] * d * (d - 1) / 2
    c1 = -(d - 2) / 3
    c2 = (1 + d) / 3

    s10 = (y[1] - y[0]) ** 2
    s11 = 13 / 12 * ((y[2] - y[1]) - (y[1] - y[0])) ** 2
    s1 = s10 + s11

    s20 = (y[2] - y[1]) ** 2
    s21 = 13 / 12 * ((y[3] - y[2]) - (y[2] - y[1])) ** 2
    s2 = s20 + s21
    a1 = c1 / (s1 + 1.e-10) ** exponent
    a2 = c2 / (s2 + 1.e-10) ** exponent
    w1 = a1 / (a1 + a2)
    w2 = a2 / (a1 + a2)
    return q1 * w1 + q2 * w2


@jit(void(float64[:, :], float64[:, :], int64[:], float64[:],
          int64, int64, float64), nopython=True)
def do_separable_y(image_rescaled, input_image, ky, dy, new_ny, nx, exponent):
    for j in range(new_ny):
        d = dy[j]
        k = ky[j]
        for i in range(nx + 2):
            image_rescaled[i, j] = scale_weno1d(d, input_image[i, k:k + 4], exponent)


@jit(void(float64[:, :], float64[:, :], int64[:], float64[:],
          int64, int64, float64), nopython=True)
def do_separable_x(image_rescaled, input_image, kx, dx, new_nx, new_ny, exponent):
    for i in range(new_nx):
        d = dx[i]
        k = kx[i]
        for j in range(new_ny):
            image_rescaled[i, j] = scale_weno1d(d, input_image[k:k + 4, j], exponent)


def conditional_decorator(func=None, *, use_extra_behavior=False):
    def decorator(original_function):
        def wrapper(*args, **kwargs):
            if use_extra_behavior:
                print("Extra behavior applied!")
            return original_function(*args, **kwargs)

        return wrapper

    # If used without parentheses, apply with default behavior
    if func is not None:
        return decorator(func)
    # If used with parentheses, return the decorator
    return decorator


def measure_time(func):
    def wrapper(*args, **kwargs):
        verbose = kwargs.pop('verbose', False)
        if verbose:
            time_start = perf_counter()
        res = func(*args, **kwargs)
        if verbose:
            time_end = perf_counter()
            print(f'Function {func.__name__} took: {time_end - time_start} s')
        return res

    return wrapper


@measure_time
def scale_nx02(orig_image, scale_exp, exponent=2., clip_every_iter=False):
    image_nx, SM, SMTotal, n_nx, n_ny, pad_offset = create_initial_scaled_image(orig_image=orig_image,
                                                                                scale=2 ** scale_exp)

    coo_slanted = np.array(create_coo_slanted())
    dx_arr_kose = create_dx(coordinates=create_coo_slanted())
    coo_straight = np.array(create_coo_ortho())
    dx_arr_ravne = create_dx(coordinates=create_coo_ortho())
    y = np.zeros(3, dtype=np.float64)

    for ex in range(scale_exp, 0, -1):
        stride = 2 ** ex
        pad_edge_image(padded_image=image_nx, offset=pad_offset)
        fill_lin_interp_padding(padded_image=image_nx, offset=pad_offset, stride=stride)

        # phase 1: slanted
        slanted_nx(coo_kose=int(stride / 2) * coo_slanted,
                   dx_arr=dx_arr_kose, exponent=np.float64(exponent),
                   n2x=np.int64(n_nx), n2y=np.int64(n_ny),
                   stride=np.int64(stride), pad_offset=np.int64(pad_offset),
                   image2x=image_nx, SM=SM, SMTotal=SMTotal, y=y)
        # phase 2: horizontal & vertical
        horz_vert_nx(coo_ravne=2 ** (ex - 1) * coo_straight, dx_arr=dx_arr_ravne, exponent=np.float64(exponent),
                     n2x=np.int64(n_nx), n2y=np.int64(n_ny),
                     stride=np.int64(stride), pad_offset=np.int64(pad_offset),
                     image2x=image_nx, SM=SM, SMTotal=SMTotal, y=y)
        if clip_every_iter:
            image_nx = np.clip(image_nx, 0, 1)

    return image_nx, pad_offset


@measure_time
def scale_separable(new_shape, input_arr, exponent):
    new_nx, new_ny = new_shape
    nx, ny = input_arr.shape

    newx = np.linspace(0, nx - 1, new_nx)
    newy = np.linspace(0, ny - 1, new_ny)

    impadshape = [(1, 1) for i in range(len(input_arr.shape))]
    image_rescaled_pad = np.pad(input_arr, impadshape, mode='reflect')

    ky = newy.astype(int)
    dy = newy - ky
    bmask_y = (ky > 0) & (dy == 0)
    ky[bmask_y] -= 1
    dy[bmask_y] = 1.

    image_rescaled_x = np.zeros((nx + 2, new_ny))
    do_separable_y(image_rescaled=image_rescaled_x,
                   input_image=image_rescaled_pad,
                   ky=ky, dy=dy, new_ny=new_ny, nx=nx, exponent=exponent)
    image_rescaled = np.zeros((new_nx, new_ny))

    kx = newx.astype(int)
    dx = newx - kx
    bmask_x = (kx > 0) & (dx == 0)
    kx[bmask_x] -= 1
    dx[bmask_x] = 1.
    do_separable_x(image_rescaled=image_rescaled, input_image=image_rescaled_x,
                   kx=kx, dx=dx, new_nx=new_nx, new_ny=new_ny, exponent=exponent)

    return image_rescaled
