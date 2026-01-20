# To run on saga:
# module load fdp
# toksearch_submit -N 1 python get_jtor_moments.py

from toksearch import MdsSignal, Pipeline


def collect_times_from_csv(csv_file):
    # Read the CSV file and return a list of dictionaries
    return [
        {"shot": shot1, "time_slices": [t0, t1]},
        {"shot": shot2, "time_slices": [t2, t3]},
        ...,
    ]


recs = collect_times_from_csv("jtor_moments.csv")

pipeline = Pipeline(recs)

# May need to play around with ordering of the dimensions
pipeline.fetch(
    "jtor", MdsSignal(r"\WHATEVER_POINTNAME", "efitrt1", dims=("times", "rho"))
)


@pipeline.map
def extract_moments(rec):
    # Extract the moments from the MDSplus signal at the given time slices
    time_slices = rec["time_slices"]
    jtor = rec["jtor"]["data"]
    times = rec["jtor"]["times"]
    rho = rec["jtor"]["rho"]

    # Whatever logic needed to get the time slices

    rec["moments"] = [m0, m1, m2] # or whatever the moments are

# Should be ok, depending on the size of the data. Let me know if it's too slow
# and we can use multiple nodes with ray
results = pipeline.compute_multiprocessing() 
