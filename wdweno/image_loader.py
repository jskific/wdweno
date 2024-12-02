from pathlib import Path
import numpy as np
from skimage import io
from skimage import util


def check_image_file(image_file: Path) -> bool:
    if not image_file.exists() or not image_file.is_file():
        raise FileNotFoundError(f"The file at '{image_file}' does not exist.")

    try:
        image = io.imread(image_file)
        _ = util.img_as_ubyte(image)
    except Exception as e:
        raise ValueError(f"The file at '{image_file}' is not a valid image: {e}")

    return True


def load_image(img_path: str, as_gray: bool = False) -> np.ndarray:
    """
    Return an image as a double array mapped from [0, 255] to [0, 1]
    """
    input_path = Path(img_path)
    input_path = input_path.absolute()

    _ = check_image_file(image_file=input_path)

    # all good?
    orig_image = io.imread(f'{str(input_path)}', as_gray=as_gray)
    orig_image = np.array(orig_image, dtype='double')
    return orig_image / 255


def load_image_deprecated(img_name: str, input_dir: str, as_gray: bool) -> np.ndarray:
    """
    Return an image as a double array mapped from [0, 255] to [0, 1]
    """
    input_path = Path(input_dir) / f'{img_name}.png'
    input_path = input_path.absolute()

    _ = check_image_file(image_file=input_path)

    # all good?
    orig_image = io.imread(f'{str(input_path)}', as_gray=as_gray)
    orig_image = np.array(orig_image, dtype='double')
    return orig_image / 255


def prepare_arr_for_img_write(in_arr: np.ndarray, round_arr: bool,
                              min_clip: float = 0., max_clip: float = 1.) -> np.ndarray:
    """
    1. in_arr is a matrix,
    2. need clip to [min_clip, max_clip]
    3. map to [0, 255]
    4. round to nearest int, if needed
    5. return array as uint8
    """
    arr = in_arr.copy()
    arr = np.clip(arr, min_clip, max_clip)
    arr *= 255
    if round_arr:
        arr = np.rint(arr)
    return arr.astype(np.uint8)


def save_image(out_file: str, img_arr: np.ndarray, round_arr: bool = True):
    out_img = prepare_arr_for_img_write(in_arr=img_arr, round_arr=round_arr)
    # out_img = img_arr
    out_path = Path(out_file)
    # with an assumption that the out path is valid
    io.imsave(out_path, out_img)
