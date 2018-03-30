# Cheminformatics tools

This repository contains various cheminformatic tools.

### Getting Started ###

Running this code requires an Anaconda distribution for Python.  This can be downloaded from the [Anaconda website](https://www.anaconda.com/download/).
You will need to add Anaconda to your path variable.  [Here is a good tutorial about installing Anaconda and adding
it to your path](https://medium.com/@GalarnykMichael/install-python-on-windows-anaconda-c63c7c3d1444).


If you have git installed, you may clone this repository by the following command:

```
git clone https://github.com/russodanielp/cheminf_tools.git
```

Alternatively, you can click the 'Clone or download' button (upper right corner) to download these files as zip.

To get started, you must first create the appropriate Python environment which is required for this project. If you have
 conda installed from the command line, this is a very simple process.  You can check by running:

 ```
 conda
 ```

 If this does not work, refer to the link about about installing Anaconda.

To create this Python environment, simply change directories to the cloned repository (or the downloaded folder containing
this project) and run the following command:

```
conda env create -f environment.yml
```

After the necessary dependecies are installed, you must activate this environment before running these scripts. To do that,
run the following command.

On windows:

```
activate cheminformatics
```

On Linux/Mac:

```
source activate cheminformatics
```

__IMPORTANT:__ This will need to be done before running any of these Python scripts.


## Building a Random Forest model

This script builds a Random Forest model from a set of training compounds contained in `.sdf` file format. It is assumed
that in the sdf there are at least two fields/columns/properties: (1) Biological activity & (2) compound names.

 It is assumed that the biological activity is already binarized, that is, 1 represents an 'active' response and 0
 represents an 'inactive' response.

 The script accepts three parameters:
 1) `-ds` -- The name of the training set `.sdf` file.
 2) `-nc` -- The name of the field/column/property that contains the compound names.
 3) `-ac` -- The name of the field/column/property that contains the biological activities.

 An example code can be run like this:
 ```
 python build_random_forest.py -ds data/CERAPP_agonist.sdf -nc Name -ac Class
 ```

After running the following files will be generated:
1) `Random_forest_5fcv_results.csv` -- `.csv` file containing five-fold cross validation scores.
2) `Random_forest_5fcv_predictions.csv` -- `.csv` file containing five-fold cross validation predictions.
3) `Random_Forest_Model.pkl` -- The saved model required for making predictions using `predict_random_forest.py`.

## Predicting a test set using a Random Forest model

This script used the Random Forest model to predict a set of test compounds in `.sdf` file format. It is assumed
that in the sdf there are at least two fields/columns/properties: (1) Biological activity & (2) compound names.  The
biological activity will be used to generate statistics.

 The script accepts three parameters:
 1) `-ts` -- The name of the test set `.sdf` file.
 2) `-nc` -- The name of the field/column/property that contains the compound names.
 3) `-ac` -- The name of the field/column/property that contains the biological activities.

 An example code can be run like this:
 ```
 python predict_random_forest.py -ds data/CERAPP_agonist.sdf -nc Name -ac Class
 ```

After running the following files will be generated:
1) `Random_forest_test_set_results.csv` -- `.csv` file containing test set scores.
2) `Random_forest_test_set_predictions.csv` -- `.csv` file containing test set predictions.