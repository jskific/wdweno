import numpy as np


def create_coo_slanted():
    dirs = [i * np.pi / 2 + np.pi / 4 for i in range(4)]
    coordinates = [[(int(np.round(np.cos(a - np.pi))), int(np.round(np.sin(a - np.pi)))),
                    (int(np.round(np.cos(a))), int(np.round(np.sin(a)))), (
                        int(np.round(np.cos(a)) + np.round(np.sqrt(5) * np.cos(a))),
                        int(np.round(np.sin(a)) + np.round(np.sqrt(5) * np.sin(a))))] for a in dirs]
    return coordinates


def create_coo_ortho():
    dirs = [i * np.pi / 2 for i in range(4)]
    coordinates = [[(int(np.round(np.cos(a - np.pi))), int(np.round(np.sin(a - np.pi)))),
                    (int(np.round(np.cos(a))), int(np.round(np.sin(a)))), (
                        int(np.round(np.cos(a)) + np.round(np.sqrt(3) * np.cos(a))),
                        int(np.round(np.sin(a)) + np.round(np.sqrt(3) * np.sin(a))))] for a in dirs]
    return coordinates


def create_dx(coordinates):
    coo = np.array(coordinates)
    axis_diff, axis_sum = 1, 2
    all_dx = np.sqrt(np.sum(np.diff(coo, n=1, axis=axis_diff) ** 2, axis=axis_sum))
    return all_dx
