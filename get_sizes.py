# To run on saga:
#
# module load toksearch
#
# You can adjust the number of nodes used by changing the -N option.
# toksearch_submit -N 3 python get_sizes.py
#
# You can adjust the list of shots and pointnames to suit your needs.

from toksearch import Pipeline
from ptdata import PtDataFetcher


def _size_of_point(pointname, shot):
    fetcher = PtDataFetcher(pointname, shot)
    header = fetcher.header
    nwords = header.nwords()
    nbytes = nwords * 2
    return nbytes

if __name__ == '__main__':

    shots = list(range(165920, 165920 + 10000))

    pipe = Pipeline(shots)

    @pipe.map
    def get_sizes(rec):
        shot = rec.shot

        pointnames = ["ip", "bt", "pcf1a", "pcf2a"]

        sizes = {}
        for pointname in pointnames:
            try:
                sizes[pointname] = _size_of_point(pointname, shot)
            except Exception as e:
                sizes[pointname] = 0

        rec["sizes"] = sizes
        rec["total_size"] = sum(sizes.values())


    results = pipe.compute_multiprocessing()

    print(f"Num results: {len(results)}")
    print(results[0])

