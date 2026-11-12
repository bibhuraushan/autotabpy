import copy
import time
import multiprocessing as mlp
from autotab.filehandle import SetPath
from autotab.datatype import datatype
from autotab.aftmap import AFTmap
from multiprocessing.pool import ThreadPool as Pool
import os
from autotab.datatype import datatype, save_aftmask
from autotab.plots import save_image

try:
    import scipy.ndimage as scnd
    import numpy as np
    import pandas as pd
    import sunpy
    import sunpy.sun as snp
    import tqdm as tq
    import h5py as h5
    import matplotlib.pyplot as plt
    from skimage.filters import gaussian, sobel
    from skimage.morphology import area_opening, binary_closing, disk, binary_erosion
    from skimage.feature import peak_local_max
    from skimage.measure import label, regionprops
    from scipy import ndimage as nd
    from astropy.stats import circmean
    from astropy import units as u
except ImportError as error:
    raise ImportError("Unable to import packages.")


def check_merging(mask1, mask2, _fluxmap, area_threshold=10):
    mask2bin = np.where(mask2 > 0, 1, 0)
    _labels = np.unique(mask1)[1:]
    for ii in _labels:
        _ind = np.where(mask1 == ii)
        overlap = mask2bin[_ind].sum()
        if overlap > area_threshold:
            _overind = np.unique(mask2[_ind])
            _overind = _overind[_overind != 0]
            if _overind.size == 1:
                continue
            else:
                minid = _overind.min()
                for jj in _overind:
                    _idx = np.where(mask2 == jj)
                    mask2[_idx] = minid
        elif (np.abs(_fluxmap[_ind]).sum() > 50) and (_ind[0].size > 1000):
            mask2[_ind] = mask2.max() + 1


def make_big_small(mask):
    # Making big regions smallar
    mask_small = np.zeros_like(mask)
    mask_large = np.zeros_like(mask)

    labels = label(np.where(mask >= 1, 1, 0), connectivity=2)
    for ii in np.unique(labels)[1:]:
        ind = np.where(labels == ii)
        if ind[0].size > 1500:
            mask_large[ind] = 1
        else:
            mask_small[ind] = 1
    mask_large = binary_erosion(mask_large, disk(3))
    _mask = mask_large + mask_small
    return _mask * mask


class TrackAR:

    def __init__(self, ardata, nthreads=10):
        self.ardata = ardata
        self.nthreads = 10
        self.currentid = 10000

    def trackin2aftmap(self, mask1, mask2, ttime, area_threshold=10, keepid=False):
        _mask1 = copy.deepcopy(mask1)
        _mask1 = make_big_small(_mask1)

        _mask2 = copy.deepcopy(mask2)
        _mask2 = make_big_small(_mask2)

        _mask11 = copy.deepcopy(mask1)
        mask1bin = np.where(_mask1 > 0, 1, 0)

        # make big region bit smaller for better tracking

        fl = open("track.log", "a")
        fl.write("-" * 20 + "\n")

        _ids = np.unique(mask2)[1:]

        for ii in _ids:
            _ind = np.where(mask2 == ii)
            _ind1 = np.where(_mask2 == ii)
            overlap = mask1bin[_ind1].sum()
            fl.write(f"For {ttime} and label {ii} overlap is {overlap}.\n")
            if overlap > area_threshold:
                _labels = np.unique(_mask11[_ind])
                _labels = _labels[_labels != 0]
                # print(np.unique(_mask11[_ind]))
                if _labels.size == 1:
                    mask2[_ind] = _labels.max()
                    mask1bin[_ind] = 0
                else:
                    minid = _labels.min()
                    mask2[_ind] = minid
                    for jj in _labels:
                        _idx = np.where(_mask1 == jj)
                        mask1[_idx] = minid
                        mask1bin[_idx] = 0
                        fl.write(f"Replacing {jj} from mask1 by {minid}.\n")
            elif not keepid:
                mask2[_ind] = self.currentid + 1
                self.currentid = self.currentid + 1
                fl.write(f"New AR found in {ttime}: Giving ID {self.currentid}.\n")
        fl.close()
        return mask2

    def get_trackedinfo(self, save_output=None, saveimage=False, restart=0):
        if os.path.exists("track.log"):
            os.remove("track.log")

        if not ((save_output is None) or (os.path.exists(save_output))):
            os.mkdir(save_output)
        trackeddata = {}

        # Start Tracking
        obj1 = next(self.ardata)
        mask1 = copy.deepcopy(obj1.mask)
        mask1 = mask1 + np.where(mask1 > 0, 1, 0) * self.currentid
        mask0 = np.zeros_like(mask1)

        self.currentid = max(mask1.max(), self.currentid)

        t0 = time.time()
        counter = 0
        for obj in self.ardata:
            counter += 1

            deltat = (time.time() - t0) / 60.0
            print(f"{obj1.Time}: For {counter} files {deltat:.3f} minutes taken.", end="\r")
            with open("log.log", "a") as fl:
                fl.write(f"{obj1.Time}: For {counter} files {deltat:.3f} minutes taken.")

            if obj.mask.max() == 0:
                with open("log.log", "a") as fl:
                    fl.write(f"{counter}. No AR found in {obj.Time}\n")
                continue

            mask2 = copy.deepcopy(obj.mask)

            check_merging(mask1, mask2, obj.fluxmap)
            _track2 = self.trackin2aftmap(mask1, mask2, obj.Time)
            trackeddata[obj1.Time] = obj1.get_labelprop(mask1)

            if not (save_output is None):
                save_aftmask(mask1, obj1, outpath=save_output)

            if not (saveimage is None):
                save_image(obj1, mask1, outpath=save_output)

            mask0 = copy.deepcopy(mask1)
            obj1 = copy.deepcopy(obj)
            mask1 = copy.deepcopy(mask2)
        trackeddata[obj1.Time] = obj1.get_labelprop(mask1)
        return datatype(trackeddata)
