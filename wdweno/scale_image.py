import numpy as np
from typing import Any

from wdweno.image_loader import check_image_file, load_image, save_image
from pathlib import Path

from wdweno.scale_nx import scale_nx02, scale_separable


def get_ndim_channel_num(orig_img: np.ndarray) -> (int, int):
    ndim = orig_img.ndim
    if ndim == 1:
        raise ValueError(f"Input image array should have ndim >=2")
    ch_num = 1 if len(orig_img.shape) == 2 else orig_img.shape[-1]
    return ndim, ch_num


def create_out_arr(target_sz: tuple, orig_img: np.ndarray) -> (np.ndarray, int):
    target_nx, target_ny = target_sz

    ndim, ch_num = get_ndim_channel_num(orig_img)
    if ndim == 2:
        out_img_arr = np.empty((target_nx, target_ny))
    else:
        out_img_arr = np.empty((target_nx, target_ny, ch_num))
    return out_img_arr, ch_num


def do_scale_with_2x_method(orig_image: np.ndarray, scale_exp: int, beta: float, verbose: bool) -> np.ndarray:
    # set out image size
    target_nx = 2 ** scale_exp * (orig_image.shape[0] - 1) + 1
    target_ny = 2 ** scale_exp * (orig_image.shape[1] - 1) + 1
    # create output img arr
    out_img_arr, ch_num = create_out_arr(target_sz=(target_nx, target_ny),
                                         orig_img=orig_image)

    for ch in range(ch_num):
        input_arr = orig_image if ch_num == 1 else orig_image[:, :, ch]

        # do the actual work
        img_new, offset = scale_nx02(orig_image=input_arr,
                                     scale_exp=scale_exp, exponent=beta,
                                     clip_every_iter=False, verbose=verbose)
        if ch_num == 1:
            out_img_arr = img_new[offset:-offset, offset:-offset]
        else:
            out_img_arr[:, :, ch] = img_new[offset:-offset, offset:-offset]

    return out_img_arr


def do_scale_with_tensor_method(orig_image: np.ndarray, scale_exp: int, beta: float, verbose: bool) -> np.ndarray:
    target_nx = 2 ** scale_exp * (orig_image.shape[0] - 1) + 1
    target_ny = 2 ** scale_exp * (orig_image.shape[1] - 1) + 1
    # create output img arr
    out_img_arr, ch_num = create_out_arr(target_sz=(target_nx, target_ny),
                                         orig_img=orig_image)

    for ch in range(ch_num):
        input_arr = orig_image if ch_num == 1 else orig_image[:, :, ch]

        # do the actual work
        img_new = scale_separable(new_shape=(target_nx, target_ny),
                                  input_arr=input_arr,
                                  exponent=beta, verbose=verbose)
        if ch_num == 1:
            out_img_arr = img_new
        else:
            out_img_arr[:, :, ch] = img_new

    return out_img_arr


def do_scale_with_free_method(orig_image: np.ndarray, scale_factor: float, beta: float, verbose: bool) -> np.ndarray:
    target_nx = int(scale_factor * orig_image.shape[0])
    target_ny = int(scale_factor * orig_image.shape[1])
    # ndim, ch_num = utils.get_ndim_channel_num(orig_img=orig_image)
    out_img_arr, ch_num = create_out_arr(target_sz=(target_nx, target_ny),
                                         orig_img=orig_image)

    scale_nums = np.maximum(1, int(np.log2(scale_factor)) + 1)

    for ch in range(ch_num):
        input_arr = orig_image if ch_num == 1 else orig_image[:, :, ch]
        img_out, offset = scale_nx02(orig_image=input_arr, scale_exp=scale_nums, exponent=beta,
                                     clip_every_iter=False)
        final_img = scale_separable(new_shape=(target_nx, target_ny),
                                    input_arr=img_out[offset:-offset, offset:-offset],
                                    exponent=beta, verbose=verbose)

        if ch_num == 1:
            out_img_arr = final_img
        else:
            out_img_arr[:, :, ch] = final_img

    return out_img_arr


def check_image(image_file: Path) -> bool:
    return check_image_file(image_file)


def check_out_image_file(image_file: Path) -> bool:
    folder = image_file.parent
    if not folder.is_dir():
        raise FileNotFoundError(f"Directory '{folder}' does not exist, is not a directory or is not valid.")

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}  # , '.gif', '.bmp', '.tiff', '.webp'}
    # Check if the file has a valid image extension
    if image_file.suffix.lower() not in valid_extensions:
        raise ValueError(f'Invalid out file extension. Not a valid image format.')

    return True


def process_kwargs(**kwargs):
    valid_methods = {'2x', 'tensor', 'free'}

    # Validate 'method'
    method = kwargs.get('method', '2x')
    if method.lower() not in valid_methods:
        raise ValueError(f"Invalid method: {method.lower()}. Must be one of {valid_methods}.")

    # Validate 'scale_exp'
    scale_exp = kwargs.get("scale_exp", 0)
    if not (isinstance(scale_exp, int) and scale_exp >= 0):
        raise ValueError(f"Invalid scale_exp: {scale_exp}. Must be an integer >= 0.")

    # Validate 'scale_factor'
    scale_factor = kwargs.get("scale_factor", 1.0)
    if not (isinstance(scale_factor, (int, float)) and scale_factor >= 0):
        raise ValueError(f"Invalid scale_factor: {scale_factor}. Must be a float >= 0.")

    # Validate 'beta'
    beta = kwargs.get("beta", 0.0)
    if not (isinstance(beta, (int, float)) and beta >= 0):
        raise ValueError(f"Invalid beta: {beta}. Must be a float >= 0.")

    # Validate 'round'
    round_arr = kwargs.get("round", True)
    if not (isinstance(round_arr, bool)):
        raise ValueError(f"Invalid round: {round_arr}. Must be True or False.")

    # Validate 'verbose'
    verbose = kwargs.get("verbose", False)
    if not (isinstance(verbose, bool)):
        raise ValueError(f"Invalid verbose: {verbose}. Must be True or False.")


    return {'method': method, 'scale_exp': scale_exp, 'scale_factor': scale_factor,
            'beta': beta, 'round_arr': round_arr, 'verbose': verbose}




def do_actual_scaling(in_image: str, out_image: str, arguments: dict[str, Any]):
    # load image
    orig_image = load_image(in_image)

    method = arguments['method']
    round_arr = arguments['round_arr']
    beta = arguments['beta']
    scale_exp = arguments['scale_exp']
    scale_factor = arguments['scale_factor']
    verbose = arguments['verbose']

    if method == '2x':
        out_img_arr = do_scale_with_2x_method(orig_image=orig_image,
                                              scale_exp=scale_exp, beta=beta, verbose=verbose)
    elif method == 'tensor':
        out_img_arr = do_scale_with_tensor_method(orig_image=orig_image,
                                                  scale_exp=scale_exp, beta=beta, verbose=verbose)
    else:  # elif method == 'free':
        out_img_arr = do_scale_with_free_method(orig_image=orig_image,
                                                scale_factor=scale_factor, beta=beta, verbose=verbose)

    save_image(out_file=out_image, img_arr=out_img_arr, round_arr=round_arr)


def scale_image(in_image: str, out_image: str, **kwargs):
    msg = f'Scale image: {Path(in_image).resolve()}'
    msg +='\n'
    msg += f'Target image: {Path(out_image).resolve()}'
    msg += '\n'
    msg += f'Supplied arguments: {kwargs}'
    msg += '\n'

    if kwargs.get("verbose", False) is True:
        print(msg)

    is_in_image_ok = check_image_file(image_file=Path(in_image).absolute())
    is_out_image_ok = check_out_image_file(image_file=Path(out_image).absolute())
    arguments = process_kwargs(**kwargs)

    if is_in_image_ok and is_out_image_ok:
        do_actual_scaling(in_image=in_image, out_image=out_image, arguments=arguments)

    return None
