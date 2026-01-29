import os
import sys
import numpy as np
import math
import argparse
from toksearch import MdsSignal, Pipeline
from toksearch_d3d import PtDataSignal
from toksearch.sql.mssql import connect_d3drdb


def downsample_to_1kHz(fetch_results: dict):
    data = fetch_results["data"]
    times = fetch_results["times"]

    # Figure out sample rate from times
    # Pick the mean of the differences between times
    # for a range of times centered around the middle
    # of the times array
    mid = len(times) // 2
    sample_rate = np.mean(np.diff(times[mid - 10 : mid + 10]))

    num_samples_per_ms = math.ceil(1 / sample_rate)

    # Make sure num_samples_per_ms is odd
    num_samples_per_ms = (
        num_samples_per_ms if num_samples_per_ms % 2 == 1 else num_samples_per_ms + 1
    )

    # Smooth the data by averaging over num_samples_per_ms

    smoothed_data = np.convolve(
        data, np.ones(num_samples_per_ms) / num_samples_per_ms, mode="valid"
    )
    trimmed_times = times[num_samples_per_ms // 2 : -num_samples_per_ms // 2 + 1]

    fetch_results["data"] = smoothed_data[::num_samples_per_ms]
    fetch_results["times"] = trimmed_times[::num_samples_per_ms]
    return fetch_results


def _build_rt_signal_dict() -> dict[str, PtDataSignal]:
    rt_sig_dict = {}
    for U_or_L in ["U", "L"]:
        for channel in range(1, 25):  # 24 channels
            power_ptname = f"DGSDPWR{U_or_L}{channel:02d}"
            rt_sig_dict[power_ptname] = PtDataSignal(power_ptname).set_callback(
                downsample_to_1kHz
            )
    return rt_sig_dict


def _build_offline_signal_dict() -> dict[str, MdsSignal]:
    offline_sig_dict = {}
    for U_or_L in ["U", "L"]:
        for channel in range(1, 25):  # 24 channels
            # Raw Bolo signals
            raw_label = f"BOL_{U_or_L}{channel:02d}_V"
            raw_pointname = f".PRAD.BOLOM.RAW:{raw_label}"
            offline_sig_dict[raw_label] = MdsSignal(
                raw_pointname, "spectroscopy"
            ).set_callback(downsample_to_1kHz)
            # pipeline.fetch(raw_label, MdsSignal(raw_pointname, "spectroscopy"))

            # Measured power
            power_label = f"BOL_{U_or_L}{channel:02d}_P"
            power_pointname = f".PRAD.BOLOM.PRAD_01:POWER.{power_label}"
            offline_sig_dict[power_label] = MdsSignal(power_pointname, "spectroscopy")
            # pipeline.fetch(power_label, MdsSignal(power_pointname, "spectroscopy"))

            # Exponential decay for convolution
            tau_label = f"{power_label}:TAU"
            tau_pointname = f".PRAD.BOLOM.PRAD_01:POWER.{tau_label}"
            offline_sig_dict[tau_label] = MdsSignal(
                tau_pointname, "spectroscopy", dims=()
            )
            # pipeline.fetch(tau_label, MdsSignal(tau_pointname, "spectroscopy", dims=()))

    return offline_sig_dict


def create_pipeline(
    num_shots: int,
    efit_tree: str = "efit01",
    savedir: str = "/mnt/beegfs/users/sammuli/for_zhe",
) -> Pipeline:

    with connect_d3drdb() as conn:
        query = """
            select top %d shot
            from shots_type
            where shot_type = 'plasma' and shot >= 193831 and shot < 195000
            order by shot desc
        """

        pipeline = Pipeline.from_sql(conn, query, num_shots)

        print(f"Processing {len(pipeline.parent._records)} shots")

    # Add signals to the pipeline

    eq_sigs_dict = {}

    # Equilibrium stuff

    # TODO: Want various scalar eq quantities, like q95, li, etc.
    # tritop, tribot, r, z, kappa, li, q95, amin, major_rad,

    # eq_sigs_dict["ssibry"] = MdsSignal(".RESULTS.GEQDSK:SSIBRY", efit_tree)
    # eq_sigs_dict["ssimag"] = MdsSignal(".RESULTS.GEQDSK:SSIMAG", efit_tree)
    # eq_sigs_dict["psirz"] = MdsSignal(
    #    ".RESULTS.GEQDSK:PSIRZ",
    #    efit_tree,
    #    dims=("r", "z", "times"),
    #    data_order=("times", "r", "z"),
    # )

    # R0, Z0, TRIBOT, TRITOP, RXPT1, ZXPT1, RXPT2, ZXPT2, KAPPA, AMINOR, DRSEP
    scalar_eq_sigs = [
        "r0",
        "z0",
        "tribot",
        "tritop",
        "rxpt1",
        "zxpt1",
        "rxpt2",
        "zxpt2",
        "kappa",
        "aminor",
        "drsep",
        "betat",
        "kappa",
        "li",
        "q95",
    ]

    for signame in scalar_eq_sigs:
        eq_sigs_dict[signame] = MdsSignal(rf"\{signame}", efit_tree)

    bolo_sig_dict = _build_rt_signal_dict()
    base_sig_name = list(bolo_sig_dict.keys())[0]

    combined_sig_dict = eq_sigs_dict | bolo_sig_dict
    pipeline.fetch_dataset("ds", combined_sig_dict)
    pipeline.align("ds", base_sig_name)

    @pipeline.map
    def trim(rec):
        ds = rec["ds"]

        ds = ds.where(ds.times > 0, drop=True)
        rec["ds"] = ds

    @pipeline.where
    def no_errors(rec):
        return not rec.errors


    @pipeline.map
    def save(rec):
        save_path = os.path.join(savedir, f"{rec.shot}.nc")
        rec["ds"].to_netcdf(save_path)
        rec["save_path"] = save_path

    pipeline.keep(["save_path"])


    return pipeline


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("num_shots", type=int)
    parser.add_argument(
        "--savedir", type=str, default="/cscratch/gahlm/output"
    )
    parser.add_argument("--efit_tree", type=str, default="efitrt1")
    parser.add_argument(
        "--backend",
        type=str,
        default="multiprocessing",
        choices=["multiprocessing", "serial"],
    )
    args = parser.parse_args()

    pipeline = create_pipeline(
        args.num_shots, efit_tree=args.efit_tree, savedir=args.savedir
    )

    if args.backend == "multiprocessing":
        results = pipeline.compute_multiprocessing()
    elif args.backend == "serial":
        results = pipeline.compute_serial()
    else:
        raise ValueError(f"Invalid backend: {args.backend}")

    print(results[0]["ds"])
    print(f"Num results: {len(results)}")
