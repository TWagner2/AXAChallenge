# AXA Challenge

Code for the AXA data science challenge.

The task is to train a classifier on CitiBike data (https://s3.amazonaws.com/tripdata/index.html) from 2018 that categorizes trips into "customers" and "subscribers".
The data is expected to be availabe in unzipped form in a folder "Data" in the working directory.


The project is grouped into several jupyter notebooks, in order:
1. **DataCleaning.ipynb**:

 Data loading, preparation and cleaning. Also some visualization of basic properties of the data. Splitting into training, validation and test set.

2. **Analysis.ipynb**:

  Visualization of the training data, focusing on differences between customers and subscribers.

3. **LogisticRegression.ipynb**:

  Training and evaluation of a logistic regression classifiers.

4. **DecisionTree.ipynb**:

  Training and evaluation of decision tree and random forest classifiers.

5. **CrashAnalysis.ipynb**:

  Analysis of NYPD Crash Data
