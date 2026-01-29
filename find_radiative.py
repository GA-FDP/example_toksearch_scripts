import pprint
import numpy as np
import argparse
from toksearch import MdsSignal, Pipeline
from toksearch_d3d import PtDataSignal

from toksearch.sql.mssql import connect_d3drdb


def create_pipeline(min_shot):
    query = """
        select shot, btorsign, ipmax, pbeam, topology
        from summaries
        where 
            shot > %d and
            btorsign < 0 and 
            ipmax > 500000 and 
            pbeam>1e6 and
            topology = 'SNB'
    """

    with connect_d3drdb() as conn:
        pipe = Pipeline.from_sql(conn, query, min_shot)

    pipe.fetch('zeff', MdsSignal(r'\zeff', 'spectroscopy'))
    pipe.fetch('prad_core', MdsSignal(r'\prad_core', 'bolom'))


    @pipe.where
    def stuff_big_enough(rec):
        zeff_big_enough = np.max(rec['zeff']['data']) > 3.
        prad_big_enough = np.max(rec['prad_core']['data']) > 5e6

        return zeff_big_enough and prad_big_enough


    @pipe.where
    def no_errors(rec):
        return not rec.errors

    return pipe

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('min_shot', type=int)

    args = parser.parse_args()


    pipe = create_pipeline(args.min_shot)


    results = pipe.compute_multiprocessing()

    print(f'NUM RESULTS: {len(results)}')


