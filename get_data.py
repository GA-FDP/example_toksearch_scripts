from toksearch import MdsSignal
from toksearch_d3d import PtDataSignal
import time

shot = 185000

start_time = time.time()
ece_res = MdsSignal(r"\tece01", "ece").fetch(shot)   #TDI expression incorrect/doesn't exist
elapsed_time = time.time() - start_time
print(f"Time elapsed fetching ECE data: {elapsed_time:.2f} seconds")
print(f"Data shape: {ece_res['data'].shape}")
print(f"Timebase shape: {ece_res['times'].shape}")

start_time = time.time()
mag_res = PtDataSignal("mpi66m322d").fetch(shot)
elapsed_time = time.time() - start_time
print(f"Time elapsed fetching magnetic data: {elapsed_time:.2f} seconds")
print(f"Data shape: {mag_res['data'].shape}")
print(f"Timebase shape: {mag_res['times'].shape}")

