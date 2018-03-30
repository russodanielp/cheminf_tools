from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, accuracy_score
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, precision_score, recall_score, confusion_matrix
from sklearn.externals import joblib

import pandas as pd

import argparse

SEED = 10

rf, params = RandomForestClassifier(class_weight='balanced',
                                  random_state=SEED), {'rf__n_estimators':[5, 10, 25, 50, 100], 'rf__max_depth':[5, 10, 15, 20]}


def make_dataset(sdf_file, nc, ac):
    mols = [mol for mol in Chem.SDMolSupplier(sdf_file) if mol]
    names = [mol.GetProp(nc) for mol in mols]
    activities = [int(mol.GetProp(ac)) for mol in mols]
    data = []

    for mol in mols:
        fcfp6 = [int(x) for x in AllChem.GetMorganFingerprintAsBitVect(mol, 3, 1024, useFeatures=True)]
        data.append(fcfp6)

    df = pd.DataFrame(data, index=names)
    df['Class'] = activities
    return df


def get_class_stats(model, X, y):
    """

    :param model: If None, assume X == y_true and y == y_pred, else should be a trained model
    :param X: Data to predict
    :param y: correct classes
    :return: 
    """
    if not model:
        predicted_classes = y
        predicted_probas = y
        y = X
    else:
        if 'predict_classes' in dir(model):
            predicted_classes = model.predict_classes(X, verbose=0)[:, 0]
            predicted_probas = model.predict_proba(X, verbose=0)[:, 0]
        else:
            predicted_classes = model.predict(X)
            predicted_probas = model.predict_proba(X)[:, 1]

    acc = accuracy_score(y, predicted_classes)
    f1_sc = f1_score(y, predicted_classes)

    # Sometimes SVM spits out probabilties with of inf
    # so set them as 1
    from numpy import inf
    predicted_probas[predicted_probas == inf] = 1

    fpr_tr, tpr_tr, thresholds_tr = roc_curve(y, predicted_probas)
    roc_auc = auc(fpr_tr, tpr_tr)
    # test classification results

    cohen_kappa = cohen_kappa_score(y, predicted_classes)
    matthews_corr = matthews_corrcoef(y, predicted_classes)
    precision = precision_score(y, predicted_classes)
    recall = recall_score(y, predicted_classes)

    # Specificity calculation
    tn, fp, fn, tp = confusion_matrix(y, predicted_classes).ravel()
    specificity = tn / (tn + fp)

    return {'ACC': acc, 'F1-Score': f1_sc, 'AUC': roc_auc, 'Cohen\'s Kappa': cohen_kappa,
            'MCC': matthews_corr, 'Precision': precision, 'Recall': recall, 'Specificity': specificity}


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Build QSAR Models')
    parser.add_argument('-ds', '--dataset', metavar='ds', type=str,
                        help='training set name')
    parser.add_argument('-nc', '--name_col', metavar='nc', type=str,
                        help='name of name column in sdf file')
    parser.add_argument('-ac', '--activity_column', metavar='ac', type=str,
                        help='name of activity column in sdf file')

    args = parser.parse_args()
    sdf_file = args.dataset
    name_col = args.name_col
    act_col = args.activity_column

    cv = StratifiedKFold(shuffle=True, n_splits=5, random_state=SEED)
    pipe = Pipeline([('rf', rf)])
    grid_search = GridSearchCV(pipe, param_grid=params, cv=cv)

    df = make_dataset(sdf_file, name_col, act_col)
    X_train, y_train_class = df.iloc[:, :-1], df['Class']

    print("Number of valid molecules: {}".format(X_train.shape[0]))
    print("Number of active compounds: {}".format((y_train_class == 1).sum()))
    print("Number of inactive compounds: {}".format((y_train_class == 0).sum()))
    print()
    grid_search.fit(X_train, y_train_class)
    best_estimator = grid_search.best_estimator_
    best_params = grid_search.best_params_
    cv_predictions = cross_val_predict(best_estimator, X_train, y_train_class, cv=cv)

    five_fold_stats = get_class_stats(None, y_train_class, cv_predictions)

    print("Finished training....")
    print("5-fold cross validation statistics:")
    for score, val in five_fold_stats.items():
        print(score, '{:.2f}%'.format(val*100))
    print()

    print("Best model parameters:")
    for param, val, in best_params.items():
        print('{}: '.format(param.replace('rf__', '')), val)

    # write 5-fold cv results to csv
    pd.Series(five_fold_stats).to_csv('Random_forest_5fcv_results.csv')
    pd.Series(cv_predictions, index=y_train_class.index).to_csv('Random_forest_5fcv_predictions.csv')

    joblib.dump(best_estimator, 'Random_Forest_Model.pkl')
