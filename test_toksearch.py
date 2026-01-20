import time
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from multiprocessing import Pool
from IPython import embed
from toksearch import Pipeline
from toksearch import MdsSignal, PtDataSignal
import MDSplus as mds



mds_server = 'atlas.gat.com'
MDSconn = mds.Connection(mds_server )


serial_shots = range(175000, 176000)
parallel_shots = range(166000, 176000)


class Timer(object):
    def __init__(self):
        self.start = None
        
    def __enter__(self):
        self.start = time.time()
        
    def __exit__(self, *args):
        elapsed = time.time() - self.start
        print('---------------------')
        print('---> Ran in {0:.2f} s'.format(elapsed))

def create_pipeline(shots):
    pipeline = Pipeline(shots)
    for ptname in ["ip", "bt"]:
        pipeline.fetch(ptname, PtDataSignal(ptname))
    return pipeline



def mds_load(arg):
    mds_server,   TDI = arg
    MDSconn = mds.Connection(mds_server )
    output = [MDSconn.get(tdi).data() for tdi in TDI]

    return output

def mds_par_load(mds_server,   TDI,  numTasks):

    #split in junks
    TDI = np.array_split(TDI, min(numTasks, len(TDI)))

    args = [(mds_server,  tdi) for tdi in TDI]

    print("pool size", len(args))

    pool = Pool(len(args))
    

    out = pool.map(mds_load,args)
    pool.close()
    pool.join()
    
    #join lists 
    return   [j for i in out for j in i]


print("*"*80)
print(f"Checking {len(serial_shots)} shots serially")
print("*"*80)

with Timer():
   print('='*30,'NAIVE SERIAL','='*30)
   for s in serial_shots:
      for n in ['ip','bt']:
          try:
              d = MDSconn.get('_x=PTDATA2("%s",%d,1);[_x, dim_of(_x)]'%(n, s) ).data()
          except:
              pass

with Timer():
    print('='*30,'TOKSEARCH SERIAL','='*30)
    pipeline = create_pipeline(serial_shots)
    serial_result = pipeline.compute_serial()
    len(serial_result)


print("*"*80)
print(f"Checking {len(parallel_shots)} shots using parallelization")
print("*"*80)

with Timer():
   print('='*30,'NAIVE PARALLEL','='*30)
   TDI =  []
   for s in parallel_shots:
      for n in ['ip','bt']:
          TDI.append('_x=PTDATA2("%s",%d,1);[_x], dim_of(_x)'%(n, s))
    

   mds_par_load(mds_server, TDI, numTasks=48)
 

print('='*30,'RAY','='*30)
# Do an initial run using ray to initialize the ray cluster
# so we're not timing the startup
dummy_pipe = Pipeline([1,2,3])
dummy_results = dummy_pipe.compute_ray()

with Timer():
    pipeline = create_pipeline(parallel_shots)
    #ray_result = pipeline.compute_ray(numparts=8)
    num_nodes = int(os.getenv("SLURM_JOB_NUM_NODES", 1))
    numparts = 2*48*num_nodes
    ray_result = pipeline.compute_ray(numparts=numparts)
    len(ray_result)


#with Timer():
#    print('='*30,'SPARK','='*30)
#    pipeline = create_pipeline(shots)
#    spark_result = pipeline.compute_spark(numparts=8)
#    len(spark_result)




    
