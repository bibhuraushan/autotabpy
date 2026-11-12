import matplotlib.pyplot as plt
import os
from autotab.functions import get_cmap
import copy
import numpy as np
from skimage.measure import regionprops

plt.style.use("autotab/bkj_style.mplstyle")


def save_image(obj, mask, outpath):
    _outpath = os.path.join(outpath, "plot")
    if not os.path.exists(_outpath):
        os.mkdir(_outpath)
    ref_aftmap = obj.filename
    newfname = os.path.basename(ref_aftmap).replace("AFTmap", "ARmap").replace("h5", "png")
    _outpath = os.path.join(_outpath, newfname)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.grid(linestyle="--", color="white", alpha=0.3)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    ax.set_xlabel("Longitude [degree]")
    ax.set_ylabel("Latitude [degree]")
    _extent = (0, 360, -90, 90)

    bmap = copy.deepcopy(obj.image)
    bmap = np.clip(bmap, -1000, 1000)
    image = (110.0 + 100.0 * np.tanh(4.0 * bmap / 1000.) / np.tanh(4.))

    try:
        _cm = get_cmap()
    except:
        _cm = "gray"

    icm = ax.imshow(image, origin="lower", extent=_extent,
                    cmap=_cm, vmax=255, vmin=0)
    ax.contour(mask, levels=np.unique(mask)[1:]-1,
               extent=_extent, linewidths=0.8)
    rprp = obj.get_labelprop(mask)
    for ii in rprp.keys():
        _lat, _lon = rprp[ii]["lat"], rprp[ii]["clon"]
        ax.text(_lon, _lat, str(int(ii)-10000),
                ha="center", fontsize=11)


    axc = fig.add_axes([0.92, 0.1, 0.025, 0.8])
    cb = plt.colorbar(icm, cax=axc, label=r"$B$ [G]")
    axc.grid(False)
    yticks = (110.0 + 100.0 * np.tanh(4.0 * np.array([-1000, -150, 0, 150, 1000]) / 1000.) / np.tanh(4.))
    axc.set_yticks(yticks)
    axc.set_yticklabels([-1000, -150, 0, 150, 1000], rotation=0, va='center')
    cb.ax.set_ylim(10, 210)
    axc.minorticks_off()
    axc.tick_params(direction="out")

    plt.savefig(_outpath, dpi=150)
    plt.close()
