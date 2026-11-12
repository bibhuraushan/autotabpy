import copy

import pandas as pd
import json
import h5py as h5
import os
import numpy as np
import yaml
from autotab.aftmap import AFTmap


def save_aftmask(mask, ref_obj, outpath="./"):
    ref_aftmap = ref_obj.filename
    newfname = os.path.basename(ref_aftmap).replace("AFTmap", "ARmap")
    year = ref_obj.Time[0:4]
    month = ref_obj.Time[5:7]
    if not os.path.exists(os.path.join(outpath,f"{year}/{month}")):
        os.makedirs(os.path.join(outpath,f"{year}/{month}"))
    outpath = os.path.join(outpath,f"{year}/{month}")            
    outfile = os.path.join(outpath, newfname)
    props = ref_obj.get_labelprop(mask)
    if os.path.exists(outfile):
        os.remove(outfile)
    with h5.File(outfile, "w") as dest:
        with h5.File(ref_aftmap, "r") as src:
            src["header"].copy(src["header"], dest)
            mapg = dest.create_group("maps")
            mapg.attrs["descr"] = "maps"
            mapg.create_dataset("armap", data=mask, dtype="int")
            dest["maps/armap"].attrs["descr"] = "AR Map"

            arsg = dest.create_group("ars")
            for ig in props.keys():
                arg = dest.create_group("ars/" + ig)
                for jj in props[ig].keys():
                    arg.create_dataset(jj, data=props[ig][jj], dtype="float")


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class datatype:
    def __init__(self, data):
        _data = copy.deepcopy(data)
        self.ID = {}
        for topl in data.keys():  # date
            for lev2 in data[topl].keys():  # ID
                new_id = self.ID.get(lev2)
                if new_id is None:
                    _lat = data[topl][lev2]["lat"]
                    _lon = data[topl][lev2]["clon"]
                    ns = lambda _lat: f"N" if _lat > 0 else f"S"
                    _ID = f"{ns(_lat)}{int(abs(_lat)):0>2}{round(_lon):0>3}"
                    new_id = topl.replace("-", "")[0:9] + _ID
                    self.ID[lev2] = new_id
                    _data[topl][new_id] = _data[topl].pop(lev2)
                else:
                    _data[topl][new_id] = _data[topl].pop(lev2)
        self.data = _data

    def to_json(self, filename, updateid=True):
        with open(filename, "w") as f:
            json.dump(self.data, f, cls=NpEncoder, indent=4)

    def to_yaml(self, filename, updateid=True):
        with open(filename, "w") as f:
            yaml.dump(self.data, f,  default_flow_style=False)

    def to_csv(self, filename, updateid=True):
        fl = open(filename, "w")
        for topl in self.data.keys():
            for lev2 in self.data[topl].keys():
                fl.write(f"{topl},{lev2}")
                for lev3 in self.data[topl][lev2]:
                    a = self.data[topl][lev2][lev3]
                    fl.write(f",{a}")
                fl.write("\n")
        fl.close()

    def to_df(self):
        self.to_csv("temp.csv")
        df = pd.read_csv("temp.csv")
        os.remove("temp.csv")
        return df
