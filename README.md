SCRIPTS
1. models.py contains the code for Model (base class), NaiveBayes, KNN and KMeans classes. It contains the logic for these classifiers.

2. io_processor.py contains the code for IO_Processor class. It contains the logic for parsing the user input via CLI.

3. learn.py contains the main function and it is the script to run via CLI.

RUN SCRIPTS
Please run learn.py as follows:
python3 learn.py -train train-file.txt [-test] [test-file.txt] [-K] [K_value] [-C] [C_value] [-d] [d_value] [centroids]

-train: the training file, required argument
-test: the test file, optional argument. NB and KNN require this argument. 
       Kmeans does not require this argument.
-K: the number of nearest neighbors, optional argument. Only use if running 
    KNN. If not given, default value is 0.
-C: Laplacian correction, optional argument. Only use if running NB. If not 
    given, default value is 0.
-d: the distance type to use, optional argument. Only use when running 
    Kmeans. Possible values: e2 or manh. 
centroids: the initial centroids, optional argument. Only use when running 
            Kmeans.

USE FOLLOWING SAMPLE COMMANDS FOR INPUT FILES PROVIDED 

KNN
python3 learn.py -train train/knn1.txt -test test/knn1.txt -K 3
python3 learn.py -train train/knn2.txt -test test/knn2.txt -K 3
python3 learn.py -train train/knn3.txt -test test/knn3.txt -K 3
python3 learn.py -train train/knn3.txt -test test/knn3.txt -K 5
python3 learn.py -train train/knn3.txt -test test/knn3.txt -K 7

NB
python3 learn.py -train train/nb1.csv -test test/nb1.csv
python3 learn.py -train train/nb1.csv -test test/nb1.csv -C 1
python3 learn.py -train train/nb2.csv -test test/nb2.csv
python3 learn.py -train train/nb2.csv -test test/nb2.csv -C 1  
python3 learn.py -train train/nb2.csv -test test/nb2.csv -C 2

KMEANS
python3 learn.py -train train/km1.txt 0,0 200,200 500,500 -d e2
python3 learn.py -train train/km1.txt 0,0 200,200 500,500 -d manh
python3 learn.py -train train/km2.txt 0,0,0 200,200,200 500,500,500 -d e2
python3 learn.py -train train/km2.txt 0,0,0 200,200,200 500,500,500 -d manh


REFERENCES
- Lecture and class notes
