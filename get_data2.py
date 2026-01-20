#!/usr/bin/env python

import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt

from scipy.io import savemat

from toksearch import PtDataSignal, Pipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('shot', type=int)
    args = parser.parse_args()

    pipe = Pipeline([args.shot])

    pointnames = ['bcoil', 'ecoil']

    for ptname in pointnames:
        pipe.fetch_dataset('ds', {ptname: PtDataSignal(ptname)})

    @pipe.where
    def no_fetch_errors(rec):
        return not rec.errors

    pipe.align('ds', 'ecoil', method='pad')

    @pipe.where
    def no_errors(rec):
        return not rec.errors
    
    res = pipe.compute_serial()

    print(len(res))
    print(res[0])

    to_save = {}
    for pointname in ['ecoil']:
        to_save[pointname] = np.concatenate([rec['ds'][pointname].values for rec in res])

    savemat('results.mat', to_save)


