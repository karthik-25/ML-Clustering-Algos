import sys

from io_processor import IO_Processor
from models import NaiveBayes, KNN, KMeans

def main():
    iop = IO_Processor()
    iop.parse_input(sys.argv[1:])

    if iop.algo == "nb":
        nb = NaiveBayes()
        nb.train(iop.train_file, iop.lap_corr)
        nb.test(iop.test_file)

    if iop.algo == "knn":
        knn = KNN()
        knn.train(iop.train_file)
        knn.test(iop.test_file, iop.k)

    if iop.algo == "kmeans":
        kmeans = KMeans()
        kmeans.train(iop.train_file, iop.centroids, iop.distance_type, iop.dim)

if __name__ == "__main__":
    main()