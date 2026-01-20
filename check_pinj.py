from toksearch import Pipeline, MdsSignal
from toksearch.sql.mssql import connect_d3drdb


if __name__ == '__main__':

    query = """
        select
            shots_type.shot as shot
        from shots_type
        where 
            shots_type.shot_type = 'plasma'
            and
            shots_type.shot >= %d
        order by NEWID()
    """
    with connect_d3drdb() as conn:
        pipeline = Pipeline.from_sql(conn, query, 110000)

    pipeline.fetch('pinj', MdsSignal(r'\pinj', 'NB'))

    @pipeline.map
    def get_shapes(rec):
        d = rec['pinj']['data']
        t = rec['pinj']['times']
        rec['d_shape'] = d.shape
        rec['t_shape'] = t.shape

    pipeline.keep(['d_shape', 't_shape'])

    @pipeline.where
    def no_errors(rec):
        return not rec.errors


    results = pipeline.compute_ray()

    print(f'NUM RESULTS: {len(results)}') 
    mismatches = [r.shot for r in results if r['d_shape'] != r['t_shape']]
    mismatches.sort()
    print(mismatches)
    print(f'NUM MISMATCHES: {len(mismatches)}')


