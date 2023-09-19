## Introduction
​	Here I use a low-dimensional dataset comprising a large number of samples, totaling 13,404 rows and 3 columns. This dataset records the luminosity of different types of seeds under varying luminous intensities. The first column represents luminous intensity, which falls within the range of 650 to 4000. The second column records the luminosity of the seeds, while the third column indicates the class to which each seed belongs. There are a total of four classes in this dataset. I have employed five different classifiers for analysis, including a fully connected neural network, decision tree, K-nearest neighbors, random forest, and support vector machine.
## Usage
  1. Run preprocess.py to preprocess the raw datasets. The same class will be merged and averaged, and the preprocessed data will be stored in `.\data\data.csv`.
  2. Run model.py and the result including csv and figure will be saved in `.\result`.

## Result

The result are presented in the form of confusion matrices. The following figure is an example.

![image-20230919174316870](C:\Users\qhd\AppData\Roaming\Typora\typora-user-images\image-20230919174316870.png){width="60%"}
