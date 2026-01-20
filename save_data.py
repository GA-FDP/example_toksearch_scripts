import argparse
from scipy.io import savemat
from toksearch import PtDataSignal, Pipeline
from toksearch.sql.mssql import connect_d3drdb

import matplotlib.pyplot as plt

def create_pipeline(max_shots=10):
    
    query = """
        select top %d shot 
        from shots_type
        where shot_type = 'plasma'
        order by newid()
    """

    with connect_d3drdb() as conn:
        pipe = Pipeline.from_sql(conn, query, 2*max_shots)


    pipe.fetch('z', PtDataSignal('vpsdfz1v'))

    @pipe.where
    def no_errors(rec):
        return not rec.errors

    return pipe

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_shots', '-N', type=int, default=10)

    args = parser.parse_args()

    max_shots = args.max_shots
    pipe = create_pipeline(max_shots=max_shots)

    print('MAX_SHOTS', max_shots)
    if max_shots <= 10:
        results = pipe.compute_serial()
    else:
        results = pipe.compute_ray()

    to_save = [dict(r) for r in results][:max_shots]

    savemat('results.mat', {'results': to_save})

    rec = to_save[0] 
    plt.plot(rec['z']['times'], rec['z']['data'])
    plt.show()
