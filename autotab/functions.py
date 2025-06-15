from matplotlib.colors import ListedColormap
from scipy.io import readsav
import copy
import numpy as np
import h5py as hdf


def get_cmap():
    red, green, blue = np.zeros(256), np.zeros(256), np.zeros(256)

    red[10:211] = np.arange(55, 256, 1)
    green[10:211] = np.arange(55, 256, 1)
    blue[10:211] = 255.0 - np.arange(0, 201, 1)

    red[255], green[255], blue[255], red[220] = 255, 255, 255, 255

    val = np.ndarray((256, 3))
    val[:, 0], val[:, 1], val[:, 2] = red, green, blue
    cm = ListedColormap(val / 255.0)

    return cm


def read_map(filepath, filetype="dat"):
    filetype = filetype.split(".")[-1]
    match filetype:
        case "dat":
            data = np.fromfile(filepath, dtype=np.float32)
            if data.size == 512 * 1024:
                bmap = data.reshape(512, -1)
            else:
                bmap = data[0:int(512 * 1024)].reshape(512, -1)
        case "sav":
            _data = readsav(filepath)
            bmap = copy.deepcopy(_data.bmap)
        case "h5":
            with hdf.File(filepath) as fl:
                bmap = np.array(fl["maps/aftmap"])
    return bmap


def get_mask(filepath):
    _data = readsav(filepath)
    mask = copy.deepcopy(_data.mask)
    return mask
