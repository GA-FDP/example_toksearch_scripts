import pprint
import numpy as np
from toksearch import Pipeline, MdsSignal, PtDataSignal

DEFAULT_CHISQ_THRESHOLD = 30

def create_pipeline(shots):

    pipe = Pipeline(shots)

    ##########################################################################
    # Get EFIT parameters
    ##########################################################################
    efit_signal_names = [
        r"\efit_a_eqdsk:betan",
        r"\efit_a_eqdsk:betap",
        r"\efit_a_eqdsk:kappa",
        r"\efit_a_eqdsk:li",
        r"\efit_a_eqdsk:gaptop",
        r"\efit_a_eqdsk:gapbot",
        r"\efit_a_eqdsk:q0",
        r"\efit_a_eqdsk:qstar",
        r"\efit_a_eqdsk:q95",
        r"\efit_a_eqdsk:wmhd",
        r"\efit_a_eqdsk:chisq",
        # r"\efit_a_eqdsk:atime", # No need for this - already retrieved
    ]

    efit_tree = "efit01"
    efit_signal_map = {
        sig_name.split(":")[-1]: MdsSignal(sig_name, efit_tree)
        for sig_name in efit_signal_names
    }

    pipe.fetch_dataset("ds", efit_signal_map)

    @pipe.map
    def filter_by_chisq(rec, chisq_thresh=DEFAULT_CHISQ_THRESHOLD):
        ds = rec["ds"]

        # Preserve this for later
        chisq = ds["chisq"]

        # Default behavior of ds.where is to replace values not
        # matching the predicate with nan
        rec["ds"] = ds.where(ds["chisq"] < chisq_thresh)
        rec["ds"]["chisq"] = chisq



    ##########################################################################
    # Get Density parameters
    ##########################################################################
    pipe.fetch_dataset("ds", {"dssdenest": PtDataSignal("dssdenest")})

    @pipe.map
    def calc_density_derivative(rec):
        ds = rec["ds"]
        ds["dssdenest_prime"] = ds["dssdenest"].differentiate("times")
        rec["ds"] = ds

    ip_signal_names = ["ipsip", "ipspr15v"]

    for sig_name in ip_signal_names:
        pipe.fetch_dataset("ds", {sig_name: PtDataSignal(sig_name)})

    @pipe.map
    def choose_ip(rec):
        ds = rec["ds"]

        if "ipsip" in ds:
            ip_sig_name = "ipsip"
        elif "ipspr15v" in ds:
            ip_sig_name = "ipspr15v"
        else:
            raise Exception("Neither ipsip or ipspr15v found")

        ds["ip"] = ds[ip_sig_name]
        ds = ds.drop(labels=["ipsip", "ipspr15v"], errors="ignore")

        rec["ds"] = ds

    pipe.fetch_dataset("ds", {"aminor_rt": MdsSignal(r"\efit_a_eqdsk:aminor", "efitrt1")})


    # Interpolate everything onto the chisq timebase
    # This probably needs to be updated. There's a bunch of things
    # we can do here.
    pipe.align("ds", "chisq", method="linear")

    @pipe.map
    def calc_greenwald(rec):
        # Need to double check units
        ds = rec["ds"]
        Ng = ds["ip"] / (np.pi * ds["aminor_rt"]**2)
        ds["Ng"] = Ng
        ds["greenwald_frac"] = ds["dssdenest"] / Ng
        rec["ds"] = ds

    return pipe

if __name__ == "__main__":

    pipe = create_pipeline([165920])

    results = pipe.compute_serial()

    pprint.pprint(results[0])





