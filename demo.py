import sys
import argparse
import numpy as np
import pprint
import socket

print(socket.gethostname())

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, freqz
from scipy.stats import pearsonr

from toksearch import Pipeline, PtDataSignal, MdsSignal
from toksearch.sql.mssql import connect_d3drdb



def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_bandpass(lower, upper, fs, order=5):
    nyq = 0.5 * fs
    normal_lower = lower /nyq
    normal_upper = upper / nyq
    b, a = butter(order, [normal_lower, normal_upper], btype='bandpass', analog=False)
    return b, a

def butter_bandpass_filter(data, lower, upper, fs, order=5):
    b, a = butter_bandpass(lower, upper, fs, order=order)
    y = filtfilt(b, a, data)
    return y

#def something():
#    pass

if __name__ == '__main__':
    print(plt.__file__)
    print(matplotlib.get_backend())
    #plt.plot([1,2,3])
    #plt.show()
    #sys.exit()

    parser = argparse.ArgumentParser()

    parser.add_argument('minshot', type=int)
    parser.add_argument('maxshot', type=int)

    args = parser.parse_args()

    query = """
        select
            shots_type.shot,
            summaries.t_ip_flat as t_start,
            summaries.t_ip_flat + summaries.ip_flat_duration as t_end
        from shots_type, summaries
        where 
            shots_type.shot_type = 'plasma' and 
            shots_type.shot = summaries.shot and 
            shots_type.shot >= %d and
            shots_type.shot <= %d
    """

    with connect_d3drdb() as conn:
        pipeline = Pipeline.from_sql(conn, query, args.minshot, args.maxshot)


    disrad_signames = ['disradu05']
    fs_signames = ['fs05']
    signames = disrad_signames + fs_signames

    disrad_sigs_dict = {signame: PtDataSignal(signame) for signame in disrad_signames}
    fs_sigs_dict = {
            signame: MdsSignal(
                fr'\{signame}',
                'spectroscopy',
                #location='remote://atlas.gat.com'
            )
            for signame in fs_signames
    }
    pipeline.fetch_dataset('ds', disrad_sigs_dict)
    pipeline.fetch_dataset('ds', fs_sigs_dict)


    @pipeline.map
    def trim(rec):
        ds = rec['ds']

        mid_time = (rec['t_start'] + rec['t_end'])/2

        t0 = mid_time - 100.0
        t1 = mid_time + 100.0

        times = ds['times']

        ds_trimmed = ds.where((times >= t0) & (times <= t1), drop=True)
        rec['ds'] = ds_trimmed
        

    #pipeline.align('ds', fs_signames[0])
    pipeline.align('ds', 0.1)
   
    @pipeline.map
    def lowpass(rec):
        ds = rec['ds']

        times = ds['times'].values / 1000.0
        
        fs = 1/np.median(np.diff(times))
        lower_cutoff = 200.0
        upper_cutoff = 1500.0

        for signame in signames:
            if upper_cutoff < fs/2:
                ds[f'{signame}_filt'] = ds[signame]*0 + butter_bandpass_filter(
                    ds[signame].values,
                    lower_cutoff,
                    upper_cutoff,
                    fs, 
                    order=5,
                 )
            else:
                ds[f'{signame}_filt'] = ds[signame]

        rec['ds'] = ds

    def filt_signame(signame):
        return f'{signame}_filt'

    @pipeline.map
    def correlate(rec):
        if rec.errors:
            pprint.pprint(rec.errors)
        ds = rec['ds']

        fs_data = ds[filt_signame(fs_signames[0])].values
        disrad_data = ds[filt_signame(disrad_signames[0])].values

        corr, _ = pearsonr(fs_data, disrad_data)
        rec['corr'] = corr

    @pipeline.where
    def non_trivial(rec):
        
        rec['abs_max'] = {}
        for signame in disrad_signames:
            abs_max = np.max(np.abs(rec['ds'][filt_signame(signame)].values))
            print(rec.shot, abs_max)
            if abs_max < 0.25:
                return False
            rec['abs_max'][signame] = abs_max
        return True
    #pipeline.keep(['abs_max', 'corr'])

    #@pipeline.where
    #def big_enough(rec):
    #    #if rec.errors:
    #    #    print('*'*80)
    #    #    print(f'SHOT: {rec.shot}')
    #    #    pprint.pprint(rec.errors)
    #    for signame in disrad_signames:
    #        abs_max = rec['abs_max'][signame]
    #        if abs_max < 0.25:
    #            print(f'shot: {rec.shot}, abs_max too small: {abs_max}')
    #            return False
    #    return True

    @pipeline.where
    def no_errors(rec):
        return not rec.errors


    results = list(pipeline.compute_ray(numparts=500))
    results.sort(key=lambda r: np.abs(r['corr']), reverse=True)

    #results = pipeline.compute_serial()


    r = results[len(results)//2]
    ds = r['ds']
    print(r)

    N = 5 
    fig, axes = plt.subplots(nrows=2, ncols=N, sharex='col')
    for i, r in enumerate(results[:N]):
        ds = r['ds']
        print(f'FOUND {len(results)} RESULTS')
        signame = disrad_signames[0]
        fsname = fs_signames[0]

        axes[0][i].plot(ds.times, ds[signame], label='orig')
        axes[0][i].plot(ds.times, ds[filt_signame(signame)], label='filt')
        axes[0][i].legend()
        axes[0][i].set_title(f'shot: {r.shot}, corr: {r["corr"]:.2f}')

        axes[1][i].plot(ds.times, ds[fsname], label='orig')
        axes[1][i].plot(ds.times, ds[filt_signame(fsname)], label='filt')

        axes[1][i].set_xlabel('time (ms)')
        if i == 0:
            axes[0][i].set_ylabel(signame)
            axes[1][i].set_ylabel(fsname)


    pprint.pprint(r.errors)

    plt.show()


