
from __future__ import division

from matplotlib import pyplot as plt
import pickle
from dataset import Trainingset, Testset
from utils import write_submission_csv
import math
from copy import deepcopy
import numpy as np

# Predictors
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier

# Misc
from xgboost import plot_importance
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, GroupKFold

# Constant
K = 10


models_preds = np.array([[],[],[],[]])
models_true = np.array([[],[],[],[]])

# colors for the plot
colors = ["blue", "green", "grey", "darkorange"]

# Initialize benchmark classifiers
classifiers = [XGBClassifier(n_estimators=1500, learning_rate=0.01, max_depth=6, subsample=0.50, 
                             colsample_bytree=0.50, colsample_bylevel=1.00, min_child_weight=2, seed=42),
                LogisticRegression(penalty='l2', C=0.5),
                RandomForestClassifier(n_estimators=500),
                SVC(C =0.4, probability=True)]

classifiers_name = ["XGBoost", "Logistic Regression", "Random Forest", "SVM"]

###############################################################################
# Find hyper parameters using Group K-Fold CV and save metrics for later plot
####################################################################################

# Cross-validation per patient 
for patient in range(1, 4):

    print "\nPatient:", patient
    print "----------\n"

    # Load training set and subsample
    trainingset = Trainingset("../Data/train_"+str(patient)+"/features/train_"+str(patient)+"_features.pgz",
                              "power_spectral_density", patient,
                              use_validation = False)

    X = trainingset.inputs
    Y = trainingset.labels
    
    # Group K-Fold
    gkfold = GroupKFold(n_splits=K)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    mean_aurocc = 0
    fold_num = 1

    # Start Cross-Validation
    for train_index, test_index in gkfold.split(X, Y, trainingset.groups):

        pred = []
        tru = []
        print "Fold :", fold_num
        fold_num += 1

        # Train all classifiers and save their predictions
        for i, classifier in enumerate(classifiers):

            print "Training classifier %d / %d" %(i+1, len(classifiers))
            classifier.fit(X[train_index], Y[train_index])
            
            plot_importance(classifier)
            plt.show()

            pred.append(classifier.predict_proba(X[test_index])[:,1])
            tru.append(Y[test_index])

        # Concatenate the predictions with previous ones
        # to compute an accuracy estimate on all folds at the end
        models_preds = np.concatenate((models_preds, pred), axis=1)
        models_true = np.concatenate((models_true, tru), axis=1)


print "Models statistics"
print "-----------------"

dump_dict = {}

for i in range(len(models_preds)):
    
    # Create a flat array with all the predictions per model
    model_true = np.ndarray.flatten(models_true[i])
    model_pred = np.ndarray.flatten(models_preds[i])

    # Compute roc curve false/true positive rates and AUROCC score
    fpr, tpr, thresholds = roc_curve(model_true, model_pred)
    aurocc = roc_auc_score(model_true, model_pred)

    # Compute precision/recall curve and scores
    precision, recall, thresholds = precision_recall_curve(model_true, model_pred)

    # Display some metrics numerically 
    print "Model :", classifiers_name[i]
    print "AUROCC :", aurocc, "  -  TPR :", np.mean(tpr), "  -  FPR :", np.mean(fpr)
    print "\n"

    dump_dict[classifiers_name[i]] = [fpr, tpr, aurocc, precision, recall]


# Dump dict (SSH) to make plot later
with open('plots.pkl', 'wb') as f:
        pickle.dump(dump_dict, f, pickle.HIGHEST_PROTOCOL)



###############################################################################
# Train classifiers with the full training set and create file for submission
####################################################################################

prediction_dict = {}

for ci, classifier in enumerate(classifiers):

    for patient in range(1, 4):

        print "Train for Kaggle submission"

        print "\nPatient:", patient
        print "----------\n"

        # Load training set and subsample
        trainingset = Trainingset("../Data/train_"+str(patient)+"/features/train_"+str(patient)+"_features.pgz",
                                "power_spectral_density", patient,
                                use_validation = False)
        
        # Load the test set
        testset = Testset("../Data/test_"+str(patient)+"_new/features/test_"+str(patient)+"_new_features.pgz",
                        "power_spectral_density")
        
        # Initialize the prediction dictionary to fill the submission file
        for file_name in testset.file_names:
            prediction_dict[file_name] = []

        X = trainingset.inputs
        Y = trainingset.labels

        # Train the classifier on patient data
        print "Training classifier %s" %(classifiers_name[ci])
        classifier.fit(X, Y)

        # Predict and fill the dictionary
        for i in range(testset.dataset_size):
        
            if len(testset.inputs[i]) > 0:
                prediction = classifier.predict_proba(testset.inputs[i])[:,1]
            else:
                prediction = np.array([0.3], dtype = np.float32)
        
            prediction_dict[testset.file_names[i]] = np.max(prediction)

    # Write predictions in submission file
    write_submission_csv(prediction_dict, "../Submissions/blank/submission.csv", 
                    "../Submissions/submission"+str(classifiers_name[ci])+".csv")