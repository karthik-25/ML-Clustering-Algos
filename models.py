class Model:
    @staticmethod
    def read_data(filename):
        data = []
        with open(filename, "r") as f:
            for line in f:
                if line == "\n":
                    continue
                line_list = line.split(",")
                data.append(line_list)

        return data

    def evaluate(self, want_list, got_list):
        label_list = sorted(list(set(want_list + got_list)))

        for label in label_list:
            TP_count = 0
            for i in range(len(want_list)):
                if want_list[i] == label and got_list[i] == label:
                    TP_count += 1

            print("Label={0} Precision={1}/{2} Recall={1}/{3}".format(label, TP_count, got_list.count(label), want_list.count(label)))

class NaiveBayes(Model):
    def train(self, train_file, lap_corr):
        self.data = self.read_data(train_file)
        self.lap_corr = lap_corr
        self.count_labels()
        self.calc_label_prob()
        self.calc_feature_dim()
        self.calc_feature_prob()

    def count_labels(self):
        self.label_count = {}
        for row in self.data:
            label = row[-1].strip()
            if label not in self.label_count:
                self.label_count[label] = 1
            else:
                self.label_count[label] += 1

    def calc_label_prob(self):
        self.label_prob = {}
        for label in self.label_count:
            self.label_prob[label] = self.label_count[label]/len(self.data)

    def calc_feature_dim(self):
        self.feature_dim = {}
        self.num_features = len(self.data[0]) - 1
        for i in range(self.num_features):
            self.feature_dim[i] = set()
            for row in self.data:
                self.feature_dim[i].add(row[i])

    def calc_feature_prob(self):
        self.num_features = len(self.data[0]) - 1
        self.label_feature_prob = {}

        for label in self.label_count:
            feature_prob_list = []
            for i in range(self.num_features):
                i_val_occurrence = {}
                for row in self.data:
                    if row[-1].strip() == label:
                        if row[i] not in i_val_occurrence:
                            i_val_occurrence[row[i]] = 1
                        else:
                            i_val_occurrence[row[i]] += 1

                for i_val, count in i_val_occurrence.items():
                    i_val_prob = (count + self.lap_corr)/(self.label_count[label] + (self.lap_corr * len(self.feature_dim[i])))
                    feature_prob_list.append((i, i_val, i_val_prob))

            self.label_feature_prob[label] = feature_prob_list
                    
    def test(self, test_file):
        want_list = []
        got_list = []
        with open(test_file, "r") as f:
            for line in f:
                line_list = line.split(",")
                want = line_list[-1].strip()
                want_list.append(want)
                got = self.predict(line_list)
                got_list.append(got)

        print("want_list={0}".format(want_list))
        print("got_list ={0}".format(got_list))
        self.evaluate(want_list, got_list)

    def predict(self, line_list):
        label_prob_list = []
        for label in self.label_count:
            label_prob = self.label_prob[label]
            for i in range(self.num_features):
                i_val = line_list[i]
                found = False
                for feature_prob in self.label_feature_prob[label]:
                    if feature_prob[0] == i and feature_prob[1] == i_val:
                        label_prob *= feature_prob[2]
                        found = True
                        break
                if not found:
                    # case when numerator is 0
                    label_prob *= self.lap_corr/(self.label_count[label] + (self.lap_corr * len(self.feature_dim[i])))

            label_prob_list.append((label_prob, label))

        return sorted(label_prob_list)[-1][1]
        

class KNN(Model):
    def train(self, train_file):
        self.data = self.read_data(train_file)

    def test(self, test_file, k):
        want_list = []
        got_list = []
        TP_count = 0
        with open(test_file, "r") as f:
            for line in f:
                n_list = []
                line_list = line.split(",")
                want = line_list[-1].strip()
                want_list.append(want)
                for row in self.data:
                    n_list.append((self.get_euclid_dist_sq(row, line_list), row[-1].strip()))

                knn_list = sorted(n_list)[:k]
                got = self.vote_and_predict(knn_list)
                got_list.append(got)
                print("want={0} got={1}".format(want, got))

        self.evaluate(want_list, got_list)

    # voting method 1: 1/d
    def vote_and_predict(self, knn_list):
        vote_dict = {}
        for n in knn_list:
            label = n[1]
            if n[0] == 0:
                weighted_d = float("inf")
            else:
                weighted_d = 1/n[0]
            if label not in vote_dict:
                vote_dict[label] = weighted_d
            else:
                vote_dict[label] += weighted_d

        # print(vote_dict)
        return max(vote_dict, key=vote_dict.get)
    
    # voting method 2: unit vote
    # def vote_and_predict(self, knn_list):
    #     vote_dict = {}
    #     for n in knn_list:
    #         label = n[1]
    #         if label not in vote_dict:
    #             vote_dict[label] = 1
    #         else:
    #             vote_dict[label] += 1

    #     # print(vote_dict)
    #     return max(vote_dict, key=vote_dict.get)
             
    def get_euclid_dist_sq(self, row, line_list):
        euclid_dist_sq = 0
        for i in range(len(line_list) - 1):
            euclid_dist_sq += (int(row[i])-int(line_list[i]))**2
        return euclid_dist_sq


class KMeans(Model):
    def train(self, train_file, centroids_list, distance_type, dim):
        self.data = self.read_data(train_file)
        self.distance_type = distance_type
        self.dim = dim
        self.points = {}
        for row in self.data:
            self.points[row[-1].strip()] = [int(x) for x in row[:-1]]
        self.centroids = {}
        for i in range(len(centroids_list)):
            self.centroids[i] = centroids_list[i]
        self.clusters = {}
        for c in self.centroids:
            self.clusters[c] = []
        
        tolerance_dist = float("inf")
        while tolerance_dist > 10**-6:
            self.print_clusters()
            print("\n")
            self.assign_clusters()
            tolerance_dist = self.recalc_centroids()

        self.print_clusters()

    def print_clusters(self):
        for cluster, pts in self.clusters.items():
            print("C{0} = {{{1}}}".format(cluster+1, ",".join(pts)))

        for c, coords in self.centroids.items():
            print("([{0}])".format(" ".join([str(x) for x in coords])))

    def recalc_centroids(self):
        tolerance_dist_list = []
        for cluster, pts in self.clusters.items():
            if len(pts) == 0:
                continue
            new_centroid = []
            for i in range(self.dim):
                sum = 0
                for pt in pts:
                    sum += self.points[pt][i]
                new_centroid.append(sum/len(pts))
            
            tolerance_dist_list.append(self.calc_tolerance(new_centroid, cluster))
            self.centroids[cluster] = new_centroid

        return max(tolerance_dist_list)

    def calc_tolerance(self, new_centroid, cluster):
        tolerance_dist = 0
        for i in range(len(new_centroid)):
            tolerance_dist += (new_centroid[i] - self.centroids[cluster][i])**2
        return tolerance_dist

    def assign_clusters(self):
        for c in self.clusters:
            self.clusters[c] = []
        
        for pt, coords in self.points.items():
            self.clusters[self.get_closest_centroid(coords)].append(pt)

    def get_closest_centroid(self, coords):
        closest_centroid = None
        closest_dist = float("inf")
        for c, vals in self.centroids.items():
            dist = self.get_distance(coords, vals)
            if dist < closest_dist:
                closest_dist = dist
                closest_centroid = c

        return closest_centroid
    
    def get_distance(self, coords, centroid_vals):
        if self.distance_type == "e2":
            dist = 0
            for i in range(len(coords)):
                dist += (int(coords[i]) - int(centroid_vals[i]))**2
            return dist
        
        if self.distance_type == "manh":
            dist = 0
            for i in range(len(coords)):
                dist += abs(int(coords[i]) - int(centroid_vals[i]))
            return dist
        