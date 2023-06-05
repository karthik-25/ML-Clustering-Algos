import sys

class IO_Processor:
    def __init__(self):
        self.input_parse_fail_str = "Error: Input parsing failed."
        self.train_file = None
        self.test_file = None
        self.k = 0
        self.lap_corr = 0
        self.distance_type = None
        self.algo = None
        self.centroids = None
        self.dim = None

    def parse_input(self, args):
        if len(args) < 1:
            print(self.input_parse_fail_str, "Invalid arguments. Sample command: python mdp.py -df .9 -tol 0.0001 some-input.txt")
            sys.exit()

        if "-train" in args:
            self.train_file = args[args.index("-train") + 1]

        if "-test" in args:
            self.test_file = args[args.index("-test") + 1]

        if "-K" in args:
            self.k = int(args[args.index("-K") + 1])

        if "-C" in args:
            self.lap_corr = int(args[args.index("-C") + 1])

        if "-d" in args:
            self.distance_type = args[args.index("-d") + 1]

        self.validate_input(args)
        
    def validate_input(self, args):
        if self.k > 0 and self.lap_corr > 0:
            print(self.input_parse_fail_str, "Cannot specify both K and Laplace Correction.")
            sys.exit()

        if self.test_file:
            if self.k == 0:
                self.algo = "nb"
            else:
                self.algo = "knn"
        else:
            if self.distance_type:
                self.algo = "kmeans"
                centroid_input = [a for a in args if "," in a]
                self.centroids = []
                for centroid in centroid_input:
                    self.centroids.append([int(c) for c in centroid.split(",")])
                self.dim = len(self.centroids[0])

            else:
                print(self.input_parse_fail_str, "Distance function required for kmeans")
                sys.exit()


