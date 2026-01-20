#!/usr/bin/env python

import numpy as np
from toksearch import PtDataSignal, Pipeline
from toksearch.sql.mssql import connect_d3drdb
from toksearch.slurm.ray_cluster import SlurmRayCluster
from toksearch.slurm.spark_cluster import SlurmSparkCluster


if __name__ == '__main__':

    query = """
        select shot 
        from shots_type 
        where shot_type = 'plasma'
        order by shot desc
    """
    with connect_d3drdb() as conn:
        pipe = Pipeline.from_sql(conn, query, batch_size=10000)

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
    

    #cluster = SlurmRayCluster.from_config()
    #cluster.start()
    #cluster.ray_init()

    cluster = SlurmSparkCluster.from_config()
    cluster.start()
    sc = cluster.spark_context()

    res = pipe.compute_spark(sc=sc, numparts=1000)

    print(len(res))
    print(res[0])
