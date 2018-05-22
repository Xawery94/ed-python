## Python projects
The repository contains two projects in Python 
* knn classifier,
* decision trees.

The project involves creating projects for the subject of data mining. The project was made using translation data
#
#### Example usage of program

`python knn.py -k 5 -m e -t train -d 5 -v iris.csv` 

- `-k INT` - number of neighbours for KNN
- `-m TYPE` - metric type (e - euclidean distance, m - manhattan distance)
- `-t TYPE` - type of test set (train, split)
- `-d INT` - (required) index of decision column
- `-v` - (optional) this flag is responsible for print verbose data
- `FILE` - (required) file path of test data

#
#### Project description:
* KNN classifier [http://www.cs.put.poznan.pl/ibladek/students/ed/lab2/projekt_knn.pdf]
* Decision trees [http://www.cs.put.poznan.pl/ibladek/students/ed/lab3/projekt_dd.pdf]

#
#### Used data for project:
* KNN classifier
    * https://archive.ics.uci.edu/ml/datasets/Iris
    * https://archive.ics.uci.edu/ml/datasets/Wine
    
* Decision trees
    * https://archive.ics.uci.edu/ml/datasets/Car+Evaluation
    * http://www.cs.put.poznan.pl/ibladek/students/ed/dane/komputery.csv
    
#
#### Dependencies
* Python 3.6 [https://www.python.org/]

#
### Creators
* Maciej Hałas
* Ksawery Janik