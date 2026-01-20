import argparse
import numpy as np
import socket
from toksearch import MdsSignal, Pipeline
from toksearch.sql.mssql import connect_d3drdb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--min-shot', type=int, default=160000)
    parser.add_argument('--min-drsep', type=float, default=0.008)
    parser.add_argument('--max-drsep', type=float, default=0.04)
    parser.add_argument('--efit-tree', default='efit01')
    parser.add_argument('--output-file', default='drsep_matching_shots.txt')

    args = parser.parse_args()

    with connect_d3drdb() as conn:
        query = """
            select shot
            from shots_type
            where shot_type = 'plasma' and shot >= %d
        """
        pipe = Pipeline.from_sql(conn, query, args.min_shot)

    pipe.fetch('drsep', MdsSignal(r'\drsep', args.efit_tree))
    
    @pipe.map
    def drsep_within_bounds(rec, min_drsep=args.min_drsep, max_drsep=args.max_drsep):
        data = rec['drsep']['data']
        within_bounds_indices = (data >= min_drsep) & (data <= max_drsep)
        rec['within_bounds_indices'] = within_bounds_indices
        rec['has_slices_within_bounds'] = np.any(within_bounds_indices)

    @pipe.where
    def has_valid_slices(rec):
        return rec['has_slices_within_bounds']

    @pipe.where
    def no_errors(rec):
        return not rec.errors

    hostname = socket.gethostname() 
    numparts = 8 if hostname.startswith('iris') else None

    results = pipe.compute_spark(numparts=numparts)
    results = list(results)
    results.sort(key=lambda x: x['shot'])
    
    with open(args.output_file, 'w') as f:
        for rec in results:
            f.write(f'{rec["shot"]}\n')
    
    print(f'NUM RESULTS: {len(results)}')
    print(f'RESULTS WRITTEN TO: {args.output_file}')
