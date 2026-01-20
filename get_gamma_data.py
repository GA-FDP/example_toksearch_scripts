import os
import glob
import pprint
import random

import numpy as np
import ray


import scipy.io as sio
import scipy.signal

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import resample

from toksearch import Signal
from toksearch import MdsSignal
from toksearch import Pipeline
from toksearch_d3d import PtDataSignal

# from tokstate.utils.utils import reduce_metrics, print_metrics, MetricAccumulator

DEFAULT_GAMMA_DIR = "/mnt/beegfs/users/sammuli/gamma_files"
MDS_LOCATION = "/mnt/beegfs/archives/mdsplus/codes/~t/~j~i/~h~g/~f~e/~d~c"

OUTPUT_DIR = "/mnt/beegfs/users/sammuli/for_kerr/data"


#class GammaDataSource(AbstractDataSource):
#    def __init__(self, directory):
#        self.directory = directory
#
#    def initialize(self, shot):
#        pass
#
#    def _gamma_file_from_shot(self, shot):
#        return os.path.join(self.directory, f"gamma_{shot}.mat")
#
#    def fetch(self, shot, pointname, **kwargs):
#        dims = kwargs.get("dims", ("times",))
#        results = {}
#
#        with open(self._gamma_file_from_shot(shot), "rb") as f:
#            matdata = sio.loadmat(f)
#
#            results["data"] = matdata["res"]["gamma"][0][0].reshape((-1,))
#
#            if not dims:
#                dims = []
#
#            if len(dims) > 0:
#                results[dims[0]] = matdata["res"]["times"][0][0].reshape((-1,))
#
#            return results
#
#    def cleanup_shot(self, shot):
#        pass
#
#    def cleanup(self):
#        pass


#def GammaSignal(directory=DEFAULT_GAMMA_DIR, **signal_kwargs):
#    return Signal("", GammaDataSource(directory), **signal_kwargs)


class GammaSignal(Signal):
    def __init__(self, directory=DEFAULT_GAMMA_DIR):
        super().__init__()

        self.directory = directory

    def _gamma_file_from_shot(self, shot):
        return os.path.join(self.directory, f"gamma_{shot}.mat")

    def gather(self, shot):

        results = {}

        with open(self._gamma_file_from_shot(shot), "rb") as f:
            matdata = sio.loadmat(f)

            results["data"] = matdata["res"]["gamma"][0][0].reshape((-1,))

            dims = self.dims
            if not dims:
                dims = []

            if len(dims) > 0:
                results[dims[0]] = matdata["res"]["times"][0][0].reshape((-1,))

            return results

    def cleanup_shot(self, shot):
        pass

    def cleanup(self):
        pass


def shot_from_filename(filename):
    """Filename of the form /path/to/dir/gamma_SHOTNO.mat"""
    return int(os.path.basename(filename).split("_")[-1].split(".")[0])


def get_available_shots(directory):
    files = glob.glob(os.path.join(directory, "gamma*.mat"))

    shots = []
    for filename in files:
        try:
            shot = shot_from_filename(filename)
            shots.append(shot)
        except Exception as e:
            print(f"Error occurred trying to parse {filename}. Moving on")
            print(e)
    return shots


def medfilt(x):
    return scipy.signal.medfilt(x, 3)


def create_pipeline(shots_dict, efit_tree="efitrt1", mode="train"):
    # shots = []
    # for key, val in shots_dict.items():
    #    shots += val

    shots = shots_dict

    pipe = Pipeline(shots)

    ########## EFIT QUANTITIES ##########################

    x_physics_ptnames = ["kappa", "li", "q95", "betan", "betap", "betat"]
    x_geom_ptnames = ["zsurf", "rsurf", "aminor"]
    x_mds_ptnames = x_physics_ptnames + x_geom_ptnames
    aux_ptnames = ["chisq", "error"]

    for ptname in x_geom_ptnames + aux_ptnames:
        sig = MdsSignal(
            r"\{}".format(ptname), efit_tree, location=MDS_LOCATION, fetch_units=False
        )
        pipe.fetch_dataset("ds", {ptname: sig})

    use_tris = True
    if use_tris:
        tri_names = ["tritop", "tribot"]
        bdry_sig = MdsSignal(
            r"\bdry",
            efit_tree,
            location=MDS_LOCATION,
            dims=("dummy1", "dummy2", "times"),
        )
        pipe.fetch("bdry", bdry_sig)

        @pipe.map
        def my_tris(rec):
            trz = rec["bdry"]["data"]

            rs = trz[:, :, 0]
            zs = trz[:, :, 1]
            r_upper = rs[:, np.argmax(zs, axis=1)].diagonal()
            r_lower = rs[:, np.argmin(zs, axis=1)].diagonal()

            ds = rec["ds"]
            ds["tritop"] = (ds.rsurf - r_upper) / (ds.aminor + 1e-6)
            ds["tribot"] = (ds.rsurf - r_lower) / (ds.aminor + 1e-6)
            rec["ds"] = ds

    else:
        tri_names = []

    for ptname in x_physics_ptnames:
        if ptname == "q95":
            which_tree = "efit01"
        else:
            which_tree = efit_tree

        sig = MdsSignal(
            r"\{}".format(ptname), which_tree, location=MDS_LOCATION, fetch_units=False
        )
        pipe.fetch_dataset("ds", {ptname: sig})

    ########## GAMMA ##########################
    pipe.fetch_dataset("ds", {"gamma": GammaSignal()})

    ########## PTDATA QUANTITIES ##########################
    ip_name = "ip"

    def convert_to_amps(d):
        d["data"] = d["data"] * 1e6
        return d

    ip_sig = PtDataSignal("ipsip", fetch_units=False).set_callback(convert_to_amps)
    pipe.fetch_dataset("ds", {ip_name: ip_sig})

    btor_name = "btor"

    def convert_bcoil_units(d):
        d["data"] = d["data"] * (2.0e-7 * 144.0 / 1.6955)
        return d 

    btor_sig = PtDataSignal(
        "pcbcoil", fetch_units=False
    ).set_callback(convert_bcoil_units)
    pipe.fetch_dataset("ds", {btor_name: btor_sig})

    flux_loop_ptnames = [
        "PCVPSI34A",
        "PCVPSI9A",
        "PCVPSI6A",
        "PCVPSI7B",
        "PCVPSI58B",
        "PCVPSI12B",
        "PCPSI1L",
        "PCPSI2L",
        "PCPSI3L",
        "PCB1L180",
    ]
    flux_loop_ptnames = []
    for ptname in flux_loop_ptnames:
        sig = PtDataSignal(ptname, fetch_units=False)
        pipe.fetch_dataset("ds", {ptname: sig})

    use_currents = False
    if use_currents:
        coils = list(range(1, 10))
        fcoil_ptdata_ptnames = [f"pcf{i}a" for i in coils] + [f"pcf{i}b" for i in coils]
        for ptname in fcoil_ptdata_ptnames:
            sig = PtDataSignal(ptname, fetch_units=False)
            pipe.fetch_dataset("ds", {ptname: sig})
    else:
        fcoil_ptdata_ptnames = []

    # ORDER IS IMPORTANT!
    x_ptnames = (
        x_mds_ptnames
        + tri_names
        + [ip_name, btor_name]
        + flux_loop_ptnames
        + fcoil_ptdata_ptnames
    )

    print("x_ptnames", x_ptnames)

    # First align to gamma
    pipe.align("ds", "gamma", method="pad")

    if mode != "test":
        # if True:
        @pipe.map
        def find_flattop(rec):
            ds = rec["ds"]
            ip = ds.ip

            b, a = scipy.signal.butter(4, 0.1, btype="lowpass")
            ip_filt = scipy.signal.filtfilt(b, a, ip) + ip * 0
            ds["ip_dot"] = ip_filt.differentiate("times") * 1e-3  # MA/s
            ds = ds.dropna("times")

            criteria = np.logical_and(
                (np.abs(ds.ip_dot) < 0.2), (ip_filt / ip_filt.max() > 0.85)
            )
            ds = ds.where(criteria, drop=True)

            rec["ds"] = ds

        # @pipe.where
        # def gamma_over_threshold(rec):
        #    return (rec['ds']['gamma'] > 100).any()

    @pipe.map
    def cleanit(rec, mode=mode):
        ds = rec["ds"]
        # ds = ds.where(ds.gamma < 500, drop=True)
        threshold = 1000
        if (mode == "train") or (mode == "val"):
            ds = ds.where(ds.gamma < threshold, drop=True)
            ds = ds.where(np.fabs(ds.ip) > 400e3, drop=True)
            ds = ds.where(ds.li > 0.01, drop=True)
            ds = ds.where(ds.kappa > 1.2, drop=True)
            ds = ds.where(ds.error < 0.1, drop=True)

        else:
            ds["gamma"] = ds["gamma"].clip(max=threshold)

        ds["gamma"] = ds["gamma"].clip(min=1e-6)
        rec["ds"] = ds

    # Now apply smoothing
    class Filt:
        def __init__(self, x_ptnames):
            self.x_ptnames = tuple(x_ptnames)

        def __call__(self, rec):
            ds = rec["ds"]
            for ptname in self.x_ptnames:
                ds[ptname].values = medfilt(ds[ptname])
            rec["ds"] = ds

    # if mode == 'train':
    if 0:
        pipe.map(Filt(x_ptnames))

    # After smoothing, THEN upsample
    if 0:

        def upsample(ds):
            orig_times = ds["times"]
            new_times = np.arange(orig_times.min(), orig_times.max(), 5.0)
            return new_times

        pipe.align("ds", upsample, method="linear")

    if 0:

        @pipe.map
        def ipdot(rec):
            ds = rec["ds"]
            ds["ipdot"] = (ds.ip - ds.ip.shift(times=1)) / (
                ds.times - ds.times.shift(times=1)
            )
            ds = ds.dropna("times")
            rec["ds"] = ds

        x_ptnames.append("ipdot")


    if 0:
        def ds_to_ndarray(ds):
            ds = ds.dropna("times")
            arr = ds.to_array().values.T
            arr = np.atleast_2d(arr).astype(np.float32)
            return arr

        @pipe.map
        def ndarrays(rec):
            rec["x"] = ds_to_ndarray(rec["x_ds"])
            rec["y"] = ds_to_ndarray(rec["y_ds"])
            rec["aux"] = ds_to_ndarray(rec["aux_ds"])
            assert rec["x"].shape[0] == rec["y"].shape[0]

        pipe.keep(["x", "y", "aux"])


    pipe.keep(["ds"])

    @pipe.where
    def no_errors(rec, debug=False):
        if debug:
            if rec.errors:
                print(rec.shot)
                pprint.pprint(rec.errors)
        return not rec.errors


    @pipe.map
    def save_to_files(rec):
        ds_to_save = rec["ds"][x_ptnames + ["gamma"]]
        file_path = os.path.join(OUTPUT_DIR, f"{rec.shot}.nc")
        ds_to_save.to_netcdf(file_path)
        rec["file_path"] = file_path
        print(file_path)
    
    pipe.keep(["file_path"])
    meta = {"x_labels": x_ptnames}
    return pipe, meta


if __name__ == "__main__":

    shots = get_available_shots(DEFAULT_GAMMA_DIR)

    random.seed(42)
    random.shuffle(shots)


    pipe, meta = create_pipeline(shots, mode="train")

    results = pipe.compute_ray()

    print(results[0])

    print(f"Number of results: {len(results)}")






