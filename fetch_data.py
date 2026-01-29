#!/usr/bin/env python

import numpy as np
from toksearch import Pipeline
from toksearch_d3d import PtDataSignal
from toksearch.sql.mssql import connect_d3drdb


if __name__ == '__main__':

    query = """
        select top 100 shot
        from shots_type
        where shot_type = 'plasma'
        order by shot desc
    """
    with connect_d3drdb() as conn:
#        pipe = Pipeline.from_sql(conn, query, batch_size=10000)
        pipe = Pipeline.from_sql(conn, query)

    pointnames = ('ip', 'ecoil', 'bcoil')
    for ptname in pointnames:
        pipe.fetch(ptname, PtDataSignal(ptname))


    @pipe.map
    def absmaxs(rec, pointnames=pointnames):
        rec['absmaxs'] = {}
        for ptname in pointnames:
            data = rec[ptname]['data']
            rec['absmaxs'][ptname] = np.max(np.abs(data))

    pipe.keep(['absmaxs'])

    @pipe.where
    def no_errors(rec):
        return not rec.errors
    

    res = pipe.compute_multiprocessing()

    print(len(res))
    print(res[0])
