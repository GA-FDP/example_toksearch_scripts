import argparse
import os

from toksearch import Pipeline, MdsSignal
from toksearch_d3d import PtDataSignal

def create_pipeline(shots):

    pipe = Pipeline(shots)

    signal_map = {
        "li": MdsSignal(r"\li", "EFIT01"),
        "betan": MdsSignal(r"\betan", "EFIT01"),
        "betat": MdsSignal(r"\betat", "EFIT01"),
        "bcentr": MdsSignal(r"\bcentr", "EFIT01"),
        "zmaxis": MdsSignal(r"\zmaxis", "EFIT01"),
        "rmaxis": MdsSignal(r"\rmaxis", "EFIT01"),
        "kappa": MdsSignal(r"\kappa", "EFIT01"),
        "q0": MdsSignal(r"\q0", "EFIT01"),
        "qstar": MdsSignal(r"\qstar", "EFIT01"),
        "q95": MdsSignal(r"\q95", "EFIT01"),
        "rq2": MdsSignal(r"\rq2", "EFIT01"),
        "zcur": MdsSignal(r"\zcur", "EFIT01"),
        "rmidin": MdsSignal(r"\rmidin", "EFIT01"),
        "rmidout": MdsSignal(r"\rmidout", "EFIT01"),
        "aminor": MdsSignal(r"\aminor", "EFIT01"),
        "area": MdsSignal(r"\area", "EFIT01"),
        "ssimag": MdsSignal(r"\ssimag", "EFIT01"),
        "ssibry": MdsSignal(r"\ssibry", "EFIT01"),
        "wmhd": MdsSignal(r"\wmhd", "EFIT01"),
        "taumhd": MdsSignal(r"\taumhd", "EFIT01"),
        "error": MdsSignal(r"\error", "EFIT01"),
        "density": MdsSignal(r"\density", "EFIT01"),
        "ipmeas": MdsSignal(r"\ipmeas", "EFIT01"),
        "gapin": MdsSignal(r"\gapin", "EFIT01"),
        "gapout": MdsSignal(r"\gapout", "EFIT01"),
        "gaptop": MdsSignal(r"\gaptop", "EFIT01"),
        "gapbot": MdsSignal(r"\gapbot", "EFIT01"),
        "ipsiptargt": PtDataSignal("ipsiptargt"),
        "ipsip": PtDataSignal("ipsip"),
        "bcoil": PtDataSignal("bcoil"),
    }


    pipe.fetch_dataset("ds", signal_map)

    @pipe.where
    def no_errors(rec):
        return not rec.errors

    return pipe

if __name__ == "__main__":
    shots = [158003, 158004, 158005, 158006, 158007, 158008, 158009, 158010]

    pipe = create_pipeline(shots)

    results = pipe.compute_serial()

    print(results[0]["ds"]['li'].dropna("times"))

    print(f"NUM RESULTS: {len(results)}/{len(shots)}")

    outdir = "/cscratch"
    os.makedirs(outdir, exist_ok=True)


    for rec in results:
        ds = rec["ds"]
        shot = rec["shot"]
        file_path = os.path.join(outdir, f"{shot}.nc")
        print(f"Writing {file_path}")
        ds.to_netcdf(file_path)


