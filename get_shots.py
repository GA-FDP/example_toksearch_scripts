import numpy as np
import random
from toksearch import Pipeline, PtDataSignal

shots = list(range(130000, 182000))
random.shuffle(shots)

pipe = Pipeline(shots)

pipe.fetch('iu30', PtDataSignal('iu30'))

@pipe.map
def max_iu30(rec):
    rec['maxval'] =  np.max(np.abs(rec['iu30']['data'])) 

@pipe.where
def above_threshold(rec):
    return rec['maxval'] > 4.5e3

@pipe.where
def no_errors(rec):
    return not rec['errors']

pipe.keep(['maxval'])

results = list(pipe.compute_ray(numparts=5000))

results.sort(reverse=True, key=lambda x: x['maxval'])

print(f'NUM SHOTS PROCESSED: {len(shots)}')
print(f'NUM MATCHING RESULTS: {len(results)}')

print('Top 10 max currents:')
for res in results[:10]:
    shot = res['shot']
    maxval = res['maxval']
    print(f'SHOT: {shot}, MAX VAL: {maxval}')
