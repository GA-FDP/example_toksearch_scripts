# To run:
# module load toksearch/latest
# python find_sweeps.py [OPTIONS]
# or
# toksearch_submit -N $NUM_NODES python -- find_sweeps.py [OPTIONS]


import argparse
import pprint
import random
import matplotlib.pyplot as plt
import numpy as np

from toksearch import Pipeline, PtDataSignal, MdsSignal
from toksearch.sql.mssql import connect_d3drdb
from toksearch.library.flattop import SimpleFlattopFinder
from toksearch.library.ell1 import L1Fit


def create_pipeline(min_shot: int = 184000, max_shot: int = 188028):

    ########################################################
    # Query for plasma shots
    ########################################################
    query = """            
        select
            shot
        from shots_type
        where
            shots_type.shot_type = 'plasma' and 
            shot >= %d and
            shot <= %d
    """

    with connect_d3drdb() as conn:
        pipe = Pipeline.from_sql(conn, query, min_shot, max_shot)

    sig_map = {
        "rvsout": MdsSignal(r"\rvsout", "efit01"),
        "ip": PtDataSignal('ip'),
    }

    pipe.fetch_dataset("ds", sig_map)

    @pipe.map
    def process_rvsout(rec):
        ########################################################
        # First trim down to use only flattop
        ########################################################
        ds = rec["ds"]
        ip_dataarray = ds['ip'].dropna('times')
        downsample = 10

        flattop_finder = SimpleFlattopFinder(
            ip_dataarray.times.values[::downsample], 
            ip_dataarray.values[::downsample],
            slope_threshold=0.05,
            lamb=10.0,
        )
        t_start = flattop_finder.t_start
        t_end = flattop_finder.t_end
        rec['t_start'] = t_start
        rec['t_end'] = t_end

        t = ds.times
        ds_trimmed = ds.where((t >= t_start) & (t <= t_end), drop=True)


        ########################################################
        # Now apply an L1 trend filter to rvsout.
        # Then, find contiguous segments from the resulting
        # fit that match our criteria (ie each segment has
        # small slope and is far enough outboard
        ########################################################
        rvsout_dataarray = ds_trimmed['rvsout'].dropna('times')

        ell1_rvsout = L1Fit(
            rvsout_dataarray.times.values,
            rvsout_dataarray.values,
            scale=float(rvsout_dataarray.max()),
            lamb=1.0,
        )

        def rvsout_seg_criteria(t_seg, d_seg):
            slope = (d_seg[-1] - d_seg[0])/(t_seg[-1] - t_seg[0])
            abs_slope = np.abs(slope) # m/ms
            min_rvsout = min(d_seg)
            return (abs_slope < 1e-4) and (min_rvsout > 1.4)


        # The group_continuous_segments method returns a list
        # of tuples. Each tuple has the start and end indices
        # of the segment (relative to the data given to the L1Fit
        # object (ell1_rvsout in this case).
        segs = ell1_rvsout.group_contiguous_segments(rvsout_seg_criteria)
        segs.sort(key=lambda seg: seg[1] - seg[0], reverse=True)

        # Just use the longest super-segment that matches our criteria
        longest_seg = segs[0]
        left_idx, right_idx = longest_seg
        rvsout_valid = rvsout_dataarray[left_idx:right_idx+1]

        rec['rvsout_valid'] = rvsout_valid 
        rec['tmin_rvsout'] = float(rvsout_valid.times[left_idx])
        rec['tmax_rvsout'] = float(rvsout_valid.times[right_idx])
        rec['min_rvsout'] = float(rvsout_valid.min())
        rec['max_rvsout'] = float(rvsout_valid.max())


    @pipe.where
    def strike_point_moves_more_than_threshold(rec, thresh=0.04):
        return (rec['max_rvsout'] - rec['min_rvsout']) >= thresh

    @pipe.where
    def no_errors(rec):
        if rec.errors:
            pprint.pprint(rec.errors)
        return not rec.errors

    return pipe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--min-shot", type=int, default=184000)

    args = parser.parse_args()

    pipe = create_pipeline(min_shot=args.min_shot)

    results = pipe.compute_ray()

    print(f"NUM RESULTS: {len(results)}")

    rec = random.choice(results)
    print("PICKING RANDOM SHOT...")
    print(f"SHOT: {rec.shot}")

    ds = rec["ds"]
    print(ds)

    to_plot = ['rvsout', 'ip']
    fig, axes = plt.subplots(nrows=len(to_plot), ncols=1, sharex=True)
    for i, name in enumerate(to_plot):
        ds[name].dropna('times').plot(ax=axes[i])
        #axes[i].axvline(rec['t_start'], color='green')
        #axes[i].axvline(rec['t_end'], color='green')

        axes[i].axvline(rec['tmin_rvsout'], color='red')
        axes[i].axvline(rec['tmax_rvsout'], color='red')
    plt.show()
