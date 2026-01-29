import numpy as np
import matplotlib.pyplot as plt
from toksearch import Pipeline
from toksearch_d3d import PtDataSignal
from toksearch.sql.mssql import connect_d3drdb



query = '''
    select shot, time_of_shot from summaries where shot = %d
'''

shot = 165920

with connect_d3drdb() as conn:
    pipe = Pipeline.from_sql(conn, query, shot)


pipe.fetch('ip', PtDataSignal('ip'))


min_time = 0
max_time = 6000.0

def _trim_signal(rec, signame, min_time, max_time):
    times = rec[signame]['times']
    data = rec[signame]['data']

    ii = np.logical_and(times > min_time, times < max_time)

    rec[signame]['times'] = times[ii]
    rec[signame]['data'] = data[ii]


@pipe.map
def trim_ip(rec):
    _trim_signal(rec, 'ip', min_time, max_time)

res = pipe.compute_serial()[0]

print(f'SHOT {res["shot"]} was at {res["time_of_shot"]}')

plt.plot(res['ip']['times'], res['ip']['data'])
plt.show()




