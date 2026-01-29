import random
import numpy as np
import csv
from toksearch import MdsSignal, Pipeline
from toksearch_d3d import PtDataSignal
from toksearch.sql.mssql import connect_d3drdb

from scipy.signal import butter, filtfilt
from scipy.signal import freqs

#import matplotlib.pyplot as plt

import pprint

def butter_lowpass(cutOff, fs, order=5):
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    b, a = butter(order, normalCutoff, btype='low', analog = False)
    return b, a

def butter_lowpass_filter(data, cutOff, fs, order=4):
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def load_disrupt_times():
    res = {}
    with open('shot_tdisrupt.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            res[int(row[0])] = 1000.*float(row[1])
    return res


def create_pipeline():
    query = """
        select shot, 1000*t_disrupt as t_disrupt
        from disruptions
        """
    with connect_d3drdb() as conn:
        pipe = Pipeline.from_sql(conn, query)

    #zsig = PtDataSignal('vpsdfz1v')
    zsig = MdsSignal(r'\zsurf', 'efit01')
    ipsig = PtDataSignal('ip')

    pipe.fetch('zraw', zsig)

    @pipe.where
    def z_exists(rec):
        return not rec.errors

    pipe.fetch('ip', ipsig)

    @pipe.map
    def smooth(rec):
        raw_data = rec['zraw']['data']
        raw_times = rec['zraw']['times']
        dt = raw_times[-1] - raw_times[-2]

        sample_freq = 1./dt
        rec['zsmooth'] = {'data': butter_lowpass_filter(raw_data,
                                                        0.1*sample_freq,
                                                        sample_freq),
                          'times': rec['zraw']['times']}

    @pipe.map
    def window_z(rec, dt=200):
        t_end = rec['t_disrupt']
        t_start = t_end - dt

        
        smooth_data = rec['zsmooth']['data']
        smooth_times = rec['zsmooth']['times']

        indices = (smooth_times > t_start) & (smooth_times < t_end)

        rec['zwindow'] = {'data': smooth_data[indices],
                          'times': smooth_times[indices]}


    @pipe.map
    def dz_dt(rec):
        z = rec['zwindow']['data']
        dz = np.diff(z)
        t = rec['zwindow']['times']
        dt = np.diff(t)
        rec['dz_dt'] = {'data': dz/(dt/1000.), 'times': t[1:]}

    @pipe.map
    def max_abs_dz_dt(rec):
        rec['max_vel'] = np.max(np.abs(rec['dz_dt']['data']))

    #pipe.keep(['max_vel'])

    @pipe.map
    def is_vde(rec):
        rec['is_vde'] = rec['max_vel'] > 2.0

    @pipe.where
    def no_errors(rec):
        return not rec.errors


    return pipe
    
if __name__ == '__main__':


    pipe = create_pipeline()

    results = list(pipe.compute_multiprocessing())

    print ("NUM RESULTS: ", len(results))
            
    velocities = [rec['max_vel'] for rec in results]

    fig, ax = plt.subplots()
    ax.hist(velocities, bins=100, log=True)
    ax.set_xlabel('Max dz_surf/dt (m/s)')
    ax.set_ylabel('count')
    plt.show()


    n_errors = 0
    n_vde = 0
    n_shots = len(results)
    for result in results:
        if result.errors:
            n_errors += 1
            continue

        if result['is_vde']:
            n_vde += 1

    print(f'NUM ERRORS: {n_errors}, NUM_VDEs: {n_vde}, NUM_SHOTS: {n_shots}')

    print(n_vde, n_shots - n_errors)

   
    
