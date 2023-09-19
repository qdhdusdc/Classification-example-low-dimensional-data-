## Introduction
  Here I use a low dimensional dataset with a large number of samples which has 13404 rows and 3 columns. It records the luminousness of different kind of seeds under different luminous intensity. The first column records luminous intensity which is in the range of 650 to 4000. The second column records the luminousness of the seed. The third column is the class of the seeds which has a total number of 4 classes.
## Usage
  1. Run preprocess.py to preprocess the raw datasets. The same class will be merged and averaged, and the preprocessed data will be stored in `.\data\data.csv`.
  2. Run model.py and the result including csv and figure will be saved in `.\result`.

