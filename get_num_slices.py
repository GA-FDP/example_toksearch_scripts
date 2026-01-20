from toksearch import MdsSignal, Pipeline

if __name__ == "__main__":

    with open("cake_shots.txt", "r") as f:
        lines = f.readlines()

    shots = [int(line) for line in lines]

    shots = shots
    pipe = Pipeline(shots)

    pipe.fetch("pt", MdsSignal(r"\ipmhd", "efit_cake02"))

    @pipe.map
    def get_num_timeslices(rec):
        data = rec["pt"]["data"]
        rec["N"] = data.shape[0]

    @pipe.where
    def no_errors(rec):
        return not rec.errors 

    pipe.keep(["N"])

    results = pipe.compute_ray()

    num_slices = 0
    for r in results:
        try:
            print(r["shot"], r["N"])
            num_slices += r["N"]
        except:
            print(r["shot"], None)

    
    print(f"{len(results)} out of {len(shots)}")
    print(f"NUM_SLICES: {num_slices}")


