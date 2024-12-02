from .scale_image import scale_image


def wdweno(in_image: str, out_image: str, **kwargs):
    """
    Main entry point for wdweno functionality.
    Wraps around the core `do_work` function.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        dict: Results from the core `do_work` function.
        :param in_image:
        :param out_image:
    """
    return scale_image(in_image=in_image, out_image=out_image, **kwargs)


# Metadata for the package
__version__ = "0.1.0"
__author__ = "Bojan Crnković, Jerko Škifić, Tina Bosner"

__all__ = ["wdweno"]
