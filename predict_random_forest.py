from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, accuracy_score
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, precision_score, recall_score, confusion_matrix
from sklearn.externals import joblib

from build_random_forest import make_dataset, get_class_stats
import os, sys

import pandas as pd

import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Predict test set using random forest')
    parser.add_argument('-ts', '--testset', metavar='ds', type=str,
                        help='test set name')
    parser.add_argument('-nc', '--name_col', metavar='nc', type=str,
                        help='name of name column in sdf file')
    parser.add_argument('-ac', '--activity_column', metavar='ac', type=str,
                        help='name of activity column in sdf file')

    args = parser.parse_args()
    sdf_file = args.testset
    name_col = args.name_col
    act_col = args.activity_column

    if not os.path.exists('Random_Forest_Model.pkl'):
        print("WARNING: No Random Forest Model found.")
        print("Please run build_random_forest.py to create a random forest model")
        sys.exit()
    else:
        best_estimator = joblib.load('Random_Forest_Model.pkl')



    df = make_dataset(sdf_file, name_col, act_col)
    X_test, y_test_class = df.iloc[:, :-1], df['Class']

    print("Number of valid molecules in test set: {}".format(X_test.shape[0]))
    print("Number of active compounds: {}".format((y_test_class == 1).sum()))
    print("Number of inactive compounds: {}".format((y_test_class == 0).sum()))
    print()

    best_estimator = joblib.load('Random_Forest_Model.pkl')
    predictions = best_estimator.predict(X_test)
    probas = best_estimator.predict_proba(X_test)

    predictions = pd.DataFrame([predictions, probas[:, 1]], index=['Prediction', 'Probability'], columns=df.index).T

    ext_set_stats = get_class_stats(best_estimator, X_test, y_test_class)

    print("Finished training....")
    print("5-fold cross validation statistics:")
    for score, val in ext_set_stats.items():
        print(score, '{:.2f}%'.format(val*100))
    print()

    # write 5-fold cv results to csv
    pd.Series(ext_set_stats).to_csv('Random_forest_test_set_results.csv')
    predictions.to_csv('Random_forest_test_set_predictions.csv')