import h5py as hdf
import matplotlib.pyplot as plt
import numpy as np
from autotab.functions import get_cmap


class AFTmap:
    """
    A class to work with AFT maps.

    Attributes:
    - nx (int): Number of pixels in the x-direction.
    - ny (int): Number of pixels in the y-direction.
    - dlat (float): Latitude step size.
    - dlatlon (float): Latitude and longitude step size.

    Methods:
    - aftmap (property): Get the AFT map.
    - mask (property): Get the mask.
    - vv (property): Get the vlat and vlon.
    - metadata (property): Get the metadata.
    - info (property): Get additional information.
    - __str__(): Return a string representation of the object.
    - view(): Display or save the AFT map visualization.
    """
    nx = 1024
    ny = 512
    dlat = np.pi / ny
    dlatlon = 2.0 * np.pi / nx

    def __init__(self, file):
        """
        Initialize an AFTmap object.

        Args:
        - file (str): Path to the HDF5 file.
        """
        self.file = file
        self.name = file.split("/")[-1]

    @property
    def aftmap(self):
        """
        Get the AFT map data.

        Returns:
        - bmap (numpy.ndarray): AFT map data.
        """
        with hdf.File(self.file) as fl:
            bmap = np.array(fl["maps/aftmap"])
        return bmap

    @property
    def mask(self):
        """
        Get the mask data.

        Returns:
        - mask (numpy.ndarray): Mask data.
        """
        with hdf.File(self.file) as fl:
            mask = np.array(fl["maps/mask"])
        return mask

    @property
    def vv(self):
        """
        Get vlat and vlon data.

        Returns:
        - vlat (numpy.ndarray): Vlat data.
        - vlon (numpy.ndarray): Vlon data.
        """
        with hdf.File(self.file) as fl:
            vlat = np.array(fl["maps/vlat"])
            vlon = np.array(fl["maps/vlon"])
        return vlat, vlon

    @property
    def metadata(self):
        """
        Get metadata.

        Returns:
        - header (dict): Metadata.
        """
        header = {}
        with hdf.File(self.file) as fl:
            for _key in fl["header"].keys():
                _val = list(fl["header"][_key])[0]
                if isinstance(_val, bytes):
                    _val = _val.decode("utf-8")
                header[_key] = _val
        return header

    @property
    def info(self):
        """
        Get additional information.

        Returns:
        - info (dict): Additional information.
        """
        info = {}
        with hdf.File(self.file) as fl:
            for _key in fl["header"].keys():
                try:
                    _info = list(fl["header"][_key].attrs["descr"])[0].decode("utf-8")
                except:
                    _info = "None"
                info[_key] = _info
        return info

    def __str__(self):
        """
        Get a string representation of the object.

        Returns:
        - str: String representation.
        """
        dct = self.metadata
        dct1 = self.info
        print("=" * 30 + f"{self.name}" + "=" * 30 + "\n")
        for _key in self.info.keys():
            print("%-10s: %-30s - %-30s" % (_key.upper(), dct[_key], dct1[_key]))
        print("-" * 85)
        self.view()
        return "=" * 85

    def view(self, para="aftmap", save=False):
        """
        Display or save the AFT map visualization.

        Args:
        - para (str, optional): Parameter to display ("aftmap" or "mask"). Defaults to "aftmap".
        - save (bool, optional): Whether to save the visualization as an image. Defaults to False.

        Returns:
        - fig (matplotlib.figure.Figure): The Figure object.
        - ax (matplotlib.axes._axes.Axes): The AxesSubplot object.
        """
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.grid(linestyle="--", color="white", alpha=0.3)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        ax.set_xlabel("Longitude [degree]")
        ax.set_ylabel("Latitude [degree]")
        _extent = (0, 360, -90, 90)

        bmap = self.aftmap
        bmap = np.clip(bmap, -1000, 1000)
        image = (110.0 + 100.0 * np.tanh(4.0 * bmap / 1000.) / np.tanh(4.))
        
        try: 
            _cm=get_cmap()
        except:
            _cm="gray"

        icm = ax.imshow(image, origin="lower", extent=_extent,
                        cmap=get_cmap(), vmax=255, vmin=0)

        axc = fig.add_axes([0.92, 0.1, 0.025, 0.8])
        cb = plt.colorbar(icm, cax=axc, label=r"$B$ [G]")
        axc.grid(False)
        yticks = (110.0 + 100.0 * np.tanh(4.0 * np.array([-1000, -150, 0, 150, 1000]) / 1000.) / np.tanh(4.))
        axc.set_yticks(yticks)
        axc.set_yticklabels([-1000, -150, 0, 150, 1000], rotation=0, va='center')
        cb.ax.set_ylim(10, 210)
        axc.minorticks_off()
        axc.tick_params(direction="out")
        if save:
            plt.savefig("data/" + "".join(self.name.split(".")[:-1]) + ".png")
            print("data/" + "".join(self.name.split(".")[:-1]) + ".png")
        else:
            plt.show()
        return fig, ax
