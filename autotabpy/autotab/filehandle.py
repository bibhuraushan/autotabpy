import os
import glob
import numpy as np


class SetPath:
    """
    This class is used to set and search file paths in a directory.
    """

    def __init__(self, path):
        """
        Constructor for SetPath class.

        Parameters:
        -----------
        path: str
            Path of the directory to be searched.
        """
        self.path = path
        self.wcard = None
        self.filelist = None
        self.counts = 0

    def search(self, wcard="*"):
        """
        Searches the directory for the specified wildcard.

        Parameters:
        -----------
        wcard: str, optional
            Wildcard pattern to search for. Default is "*".

        Returns:
        --------
        _list: numpy array
            List of files found in the directory.
        """
        files = []
        dirs = []
        for _root, _dirs, _ in os.walk(self.path):
            fls = glob.glob(wcard, root_dir=_root)
            if len(fls) == 0:
                continue
            for fl in fls:
                files.append(os.path.join(_root, fl))
        _list = np.sort(np.array(files))
        self.filelist = _list
        self.wcard = wcard
        self.counts = _list.size
        return _list

    @property
    def basename(self):
        """
        Extracts the basename of the files in the directory.

        Returns:
        --------
        name: numpy array
            List of basenames of the files found in the directory.
        """
        name = []
        array = self.search(wcard=self.wcard)
        for _name in array:
            name.append(os.path.basename(_name))
        return np.array(name)
