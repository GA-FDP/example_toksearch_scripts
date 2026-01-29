from toksearch import Pipeline, MdsSignal

OUTPUT_DIR = "/cscratch/output"


def no_errors(rec):
    return not rec.errors

if __name__ == '__main__':

    first_shot = 165920
    shots = list(range(first_shot, first_shot + 10000))

    pipeline = Pipeline(shots)
   
    dims = ['r', 'z', 'times']
    psirz_sig = MdsSignal(r'\psirz', 'efit01', dims=dims, data_order=['times', 'r', 'z'])

    pipeline.fetch_dataset("ds", {"psirz": psirz_sig})

    pipeline.where(no_errors)
    @pipeline.map
    def write_ds(rec):
        ds = rec["ds"]
        ds.to_netcdf(f"{OUTPUT_DIR}/{rec.shot}.nc")

    pipeline.where(no_errors)

    pipeline.keep([])

    results = pipeline.compute_multiprocessing(num_workers=8)

    print(results[0])
    print(f"Number of results: {len(results)}")
