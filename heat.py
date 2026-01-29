import sys
import os
import argparse
import glob
import pprint
import numpy as np
import xarray as xr
from toksearch import Pipeline, MdsSignal
from toksearch_d3d import PtDataSignal

from toksearch.sql.mssql import connect_d3drdb

def create_pipeline(args):
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
    min_shot = args.min_shot
    max_shot = args.max_shot
    with connect_d3drdb() as conn:
        pipeline = Pipeline.from_sql(conn, query, min_shot, max_shot)


    ##############################################################################
    # Scalar signals
    ##############################################################################
    scalar_inputs = {}

    scalar_inputs['ip'] = MdsSignal(r'\ipmeas', 'efit01')
    scalar_inputs['density'] = MdsSignal(r'\density', 'efit01')
    scalar_inputs['kappa'] = MdsSignal(r'\kappa', 'efit01')
    #scalar_inputs['prad_core'] = MdsSignal(r'\prad_core', 'BOLOM')
    #scalar_inputs['prad_divl'] = MdsSignal(r'\prad_divl', 'BOLOM')
    scalar_inputs['irtvpopr2'] = MdsSignal(r'\irtvpopr2', 'IRTV')
    scalar_inputs['irtvpipr2'] = MdsSignal(r'\irtvpipr2', 'IRTV')

    scalar_inputs['bcoil'] = PtDataSignal('bcoil')
    scalar_inputs['vloop'] = PtDataSignal('vloop')

    scalar_inputs['rvsout'] = MdsSignal(r'\rvsout', 'efit01')
    scalar_inputs['zvsout'] = MdsSignal(r'\zvsout', 'efit01')
    scalar_inputs['rvsin'] = MdsSignal(r'\rvsin', 'efit01')
    scalar_inputs['zvsin'] = MdsSignal(r'\zvsin', 'efit01')
    scalar_inputs['tritop'] = MdsSignal(r'\tritop', 'efit01')
    scalar_inputs['tribot'] = MdsSignal(r'\tribot', 'efit01')

    scalar_inputs['spred_O5c'] = MdsSignal(r'\spred_O5c', 'SPECTROSCOPY')
    scalar_inputs['spred_c3'] = MdsSignal(r'\spred_c3', 'SPECTROSCOPY')
    scalar_inputs['spred_cu19'] = MdsSignal(r'\spred_cu19', 'SPECTROSCOPY')
    scalar_inputs['spred_Fe23'] = MdsSignal(r'\spred_Fe23', 'SPECTROSCOPY')
    scalar_inputs['spred_ni26'] = MdsSignal(r'\spred_ni26', 'SPECTROSCOPY')
    scalar_inputs['spred_mo31'] = MdsSignal(r'\spred_mo31', 'SPECTROSCOPY')

    ##############################################################################
    # Scalar signals
    ##############################################################################
    power_inputs = {}
    power_inputs['pinj'] = MdsSignal(r'\pinj', 'NB')

    def drop_last(rec):
        rec['data'] = rec['data'][:-100]
        rec['times'] = rec['times'][:-100]
        return rec

    power_inputs['echpwrc'] = MdsSignal(r'\echpwrc', 'd3d')
    power_inputs['poh'] = MdsSignal(r'\poh', 'd3d')

    ##############################################################################
    # Profile signals
    ##############################################################################
    #profile_inputs = {}
    #profile_inputs['qdep'] = MdsSignal(
    #    r'\WALL::TOP.IRTV.IRTV:LODIV_60RP2:DIGITAL_CAM:HEAT_FLUX2D',
    #    'wall',
    #    dims=('r', 'times'),
    #    data_order=('times', 'r'),
    #)

    #profile_inputs['tdep'] = MdsSignal(
    #    r'\WALL::TOP.IRTV.IRTV:LODIV_60RP2:DIGITAL_CAM:ABS_T_2D',
    #    'wall',
    #    dims=('r', 'times'),
    #    data_order=('times', 'r'),
    #)

    ##############################################################################
    # 2D signals
    ##############################################################################
    #grid_inputs = {}
    #grid_inputs['psirz'] = MdsSignal(
    #    r'\psirz',
    #    'efit01',
    #    dims=('R', 'Z', 'times'),
    #    data_order=('times', 'R', 'Z'),
    #)


    pipeline.fetch_dataset('ds', scalar_inputs)


    @pipeline.where
    def no_ds_errors(rec):
        return not rec.errors

    pipeline.fetch_dataset('ds_power', power_inputs)


    @pipeline.map
    def fill_in_missing_pwr(rec):
        #reset errors
        rec.errors = {}

        ds = rec['ds']
        if 'ds_power' not in rec:
            rec['ds_power'] = {}
        ds_power = rec['ds_power']
        for signame in ['echpwrc', 'pinj', 'poh']:
            try:
                if signame not in ds_power:
                    ds[signame] = ds['ip']*0
                else:
                    ds = xr.merge(
                        [ds, ds_power[signame].to_dataset(name=signame)]
                    )
            except Exception as e:
                print(e)
                raise(e)

        rec['ds'] = ds

    pipeline.keep(['ds', 't_start', 't_end'])

    # Do alignment prior to fetching psirz to conserve memory
    if not args.no_align:
        pipeline.align('ds', 'ip', method='pad')


    @pipeline.map
    def trim(rec):
        ds = rec['ds']
        t0 = rec['t_start']
        t1 = rec['t_end']
        times = ds['times']
        ds_trimmed = ds.where((times >= t0) & (times <= t1), drop=True)
        rec['ds'] = ds_trimmed


    @pipeline.where
    def is_lower_single_null(rec):
        ds = rec['ds']
        epsilon = 1e-6
        tri_ratio = ds['tribot']/(ds['tritop'] + epsilon)
        median_tri_ratio = np.median(tri_ratio.values)
        rec['median_tri_ratio'] = median_tri_ratio

        return median_tri_ratio > 2.0


    @pipeline.map
    def save_dataset(rec, output_dir=args.output_dir):
        ds = rec['ds']
        shot = rec['shot']
        path = os.path.join(output_dir, f'shot_{shot}.nc')
        ds.to_netcdf(path)
        rec['path'] = path

    
    pipeline.keep(['ds', 'path']) 


    @pipeline.where
    def no_errors(rec, debug=False):
        if debug and rec.errors:
            pprint.pprint(rec.errors)
        return not rec.errors

    return pipeline


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('min_shot', type=int)
    parser.add_argument('max_shot', type=int)
    parser.add_argument('--no-align', action='store_true')
    parser.add_argument('--serial', action='store_true')
    parser.add_argument('--output-dir', default='data')

    args = parser.parse_args()

    N = args.max_shot - args.min_shot


    existing_files = glob.glob(os.path.join(args.output_dir, 'shot_*.nc'))
    print(f'Removing {len(existing_files)} from {args.output_dir}')
    for f in existing_files:
        os.remove(f)


    pipeline = create_pipeline(args)
    if args.serial:
        results = pipeline.compute_serial()
    else:
        results = pipeline.compute_multiprocessing()

       
    print(f'FOUND {len(results)} RESULTS')
    print(results[-1].shot)
    pprint.pprint(results[-1].errors)
    print(results[-1]['ds'])

    max_echs = [float(r['ds']['echpwrc'].values.max()) for r in results]

    max_ech = max(max_echs) 
    print(f'Max ech: {max_ech}')


    sys.exit()


    print(ds)
    print('nbytes', ds.nbytes)

    tdep = r['ds']['tdep'].dropna('times').drop('times').drop('r').values
    print(tdep.shape)




