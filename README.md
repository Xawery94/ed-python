## Python projects
The repository contains two projects in Python 
* knn classifier,
* decision trees.

The project involves creating projects for the subject of data mining. The project was made using translation data
#
#### Example usage of program for KNN

`python knn.py -k 5 -m e -t train -d 5 -v iris.csv` 

- `-k INT` - number of neighbours for KNN
- `-m TYPE` - metric type (e - euclidean distance, m - manhattan distance)
- `-t TYPE` - type of test set (train, split)
- `-s FLOAT` - size of split
- `-d INT` - (required) index of decision column
- `-v` - (optional) this flag is responsible for print verbose data
- `FILE` - (required) file path of test data

#
#### Additional steps for tree program

- Install graphviz for python
```bash
pip install graphviz
```

* Next steps
* For Mac:
```bash
brew install graphviz
```
* For Windows:
    - Install from [https://graphviz.gitlab.io/_pages/Download/Download_windows.html]
    - Set PAHT to `C:/Program Files (x86)/GraphvizX.XX/bin/`
    - Reopen user editor

#
#### Example usage of program for Decision Tree

`python tree.py -l 3 -a 2 -i 0.5 car.csv` 

- `-l INT` - decision column
- `-a INT` - amount of decision classes
- `-i FLOAT` - minimal infoGain for attribute type (default 0.0)
- `-ih` - enable header from csv
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
* Graphviz [https://graphviz.gitlab.io/download/]

#
### Creators
* Maciej Hałas
* Ksawery Janik