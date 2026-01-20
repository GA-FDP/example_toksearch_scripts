from toksearch import Pipeline
import torch


pipe = Pipeline([1,2,3,4])


results = pipe.compute_ray()

available = torch.cuda.is_available()

print(f"IS AVAILABLE: {available}")



