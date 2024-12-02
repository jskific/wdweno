# import matplotlib.pyplot as plt
import numpy as np
import skimage.transform


def init_arrays(orig_image):
    """
    Traba mi numpy i skimage.transform za ovo
    """
    im_shape = np.array(orig_image.shape)
    impadshape = [(3, 3) for i in range(len(im_shape))]
    image = np.pad(orig_image, impadshape, mode='reflect')
    # image = np.pad(orig_image, impadshape, mode='constant', constant_values=1)
    # image = np.pad(orig_image, impadshape, mode='edge')
    nx, ny = np.shape(image)
    n2x = 2 * nx - 1
    n2y = 2 * ny - 1
    image_rescaled_pad = skimage.transform.resize(image, (n2x, n2y), order=3, mode='reflect')

    image2x = np.zeros((n2x, n2y))

    SMTotald0 = np.zeros_like(image2x)
    SMTotald1 = np.zeros_like(image2x)

    SM = np.zeros_like(image2x)
    SMTotal = np.zeros_like(image2x)

    # fill from the original image
    image2x[::2, ::2] = image

    # fill padding
    image2x[0:6, :] = image_rescaled_pad[0:6, :]
    image2x[-6:, :] = image_rescaled_pad[-6:, :]
    image2x[:, 0:6] = image_rescaled_pad[:, 0:6]
    image2x[:, -6:] = image_rescaled_pad[:, -6:]

    return image2x, SM, SMTotal, SMTotald0, SMTotald1, n2x, n2y


def init_nscale_arrays(orig_image, scale_exp, pad_sz):
    im_shape = np.array(orig_image.shape)
    scale = 2 ** scale_exp

    impadshape = [(pad_sz, pad_sz) for i in range(len(im_shape))]
    image = np.pad(orig_image, impadshape, mode='reflect')

    nx, ny = np.shape(image)
    n_nx = scale * (nx - 1) + 1
    n_ny = scale * (ny - 1) + 1

    image_rescaled_pad = skimage.transform.resize(image, (n_nx, n_ny), order=3, mode='reflect')
    image_nx = np.zeros((n_nx, n_ny))

    # fill from the original image
    image_nx[::scale, ::scale] = image

    offset = scale * pad_sz
    # fill padding
    image_nx[0:offset, :] = image_rescaled_pad[0:offset, :]
    image_nx[-offset:, :] = image_rescaled_pad[-offset:, :]
    image_nx[:, 0:offset] = image_rescaled_pad[:, 0:offset]
    image_nx[:, -offset:] = image_rescaled_pad[:, -offset:]

    SM = np.zeros_like(image_nx)
    SMTotal = np.zeros_like(image_nx)

    return image_nx, SM, SMTotal, n_nx, n_ny, offset


def create_initial_scaled_image(orig_image, scale):
    """
    Create an array for scaled image with added empty ghost points
    :return: an array for scaled image and offset
    """
    pad_sz = 3
    nx, ny = np.shape(orig_image)
    n_nx = scale * (nx - 1) + 1
    n_ny = scale * (ny - 1) + 1

    # image_nx = np.zeros((n_nx, n_ny))
    # fill from the original image
    # image_nx[::scale, ::scale] = orig_image
    offset = pad_sz * scale
    padded_image = np.zeros((n_nx + 2 * offset, n_ny + 2 * offset))
    padded_image[offset:-offset:scale, offset:-offset:scale] = orig_image

    SM = np.zeros_like(padded_image)
    SMTotal = np.zeros_like(padded_image)

    total_x, total_y = padded_image.shape

    return padded_image, SM, SMTotal, total_x, total_y, offset


def pad_edge_image(padded_image, offset):
    """
    Similar to numpy.pad with mode='edge'.
    The image is already padded, we just fill the ghost points with the edge points.
    """
    # left
    padded_image[offset:-offset, :offset] = padded_image[offset:-offset, offset][:, None]

    # right
    padded_image[offset:-offset, -offset:] = padded_image[offset:-offset, -offset - 1][:, None]

    # up
    padded_image[0, offset:-offset] = padded_image[offset, offset:-offset]

    # down
    padded_image[-1, offset:-offset] = padded_image[-offset - 1, offset:-offset]

    # up left:
    padded_image[0:offset, 0:offset] = padded_image[offset, offset]
    # up right:
    padded_image[0:offset, -offset:] = padded_image[offset, -offset - 1]
    # down left:
    padded_image[-offset:, 0:offset] = padded_image[-offset - 1, offset]
    # down right
    padded_image[-offset:, -offset:] = padded_image[-offset - 1, -offset - 1]


def fill_lin_interp_padding(padded_image, offset, stride):
    """
    Empty ghost points are between padded need to be interpolated with
    """
    dx = 1.
    y = padded_image[offset:-offset:stride, 0]
    yinterp = y[1:] + (y[1:] - y[:-1]) / 2
    padded_image[offset + int(stride / 2):-offset - int(stride / 2):stride, :offset] = yinterp[:, None]

    y = padded_image[0, offset:-offset:stride]
    yinterp = y[1:] + (y[1:] - y[:-1]) / 2
    padded_image[:offset, offset + int(stride / 2):-offset - int(stride / 2):stride] = yinterp

    y = padded_image[offset:-offset:stride, -1]
    yinterp = y[1:] + (y[1:] - y[:-1]) / 2
    padded_image[offset + int(stride / 2):-offset - int(stride / 2):stride, -offset:] = yinterp[:, None]

    y = padded_image[-1, offset:-offset:stride]
    yinterp = y[1:] + (y[1:] - y[:-1]) / 2
    padded_image[-offset:, offset + int(stride / 2):-offset - int(stride / 2):stride] = yinterp
