
# coding: utf-8

get_ipython().system(u' pip install -U scikit-learn')

print ("\n~~ done installing ~~\n")

from __future__ import division
import warnings

from math import log
import numpy as np

from scipy.optimize import fmin_bfgs
from sklearn.preprocessing import LabelEncoder

print ("\n~~ done importing~~\n")


# load modules
from __future__ import division
import pandas as pd
pd.set_option('precision', 3)
import numpy as np
import matplotlib.pylab as plt
get_ipython().magic(u'matplotlib inline')
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.calibration import CalibratedClassifierCV
import seaborn as sns
from scipy import interp
import bisect
from scipy.stats import mstats
#from churn_measurements import calibration, discrimination



df = pd.read_csv('C:/Users/MoideenJ/Desktop/TelcoChurn/InputChurnData.csv',low_memory=False, header=0)

print ("~~ Read Input Data ~~")



col_names = df.columns.tolist()
print ("\n Feature Column Names:\n")
print (col_names)



to_show = col_names[:10] + col_names[-10:]
print ('\n Sample Input Data:')
df[to_show].head(10)



print ("\n~~ Input Data Summary ~~\n")
print (df.shape)
print (df.dtypes)



# The number of numeric data
len(df.select_dtypes(include=['int64','float64']).columns)



# The number of categorical data
len(df.select_dtypes(include=['category','object']).columns)



# Check there is any missing data
for i in df.columns.tolist():
    k = sum(pd.isnull(df[i]))
    print (i, k)



print "~ Exploring Data ~"
# numeric data
#print df.describe()
# same as above
print df.describe(include=['int64','float64'])



# categorical and object data
print df.describe(include=['category','object'])



print df['HAS_CHURNED'].value_counts()



print df.groupby(df['HAS_CHURNED']).mean()



# Histogram
df.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(df.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
ticks = np.arange(0, len(df._get_numeric_data().columns), 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(col_names)
ax.set_yticklabels(col_names)
plt.show()


print "~ Preparing Target and Features Data ~"

# Isolate target data
y = np.where(df['HAS_CHURNED']== 'Yes', 1, 0)

print(y)

print "~ Feature Converted to Boolean ~"

# 'yes'/'no' has to be converted to boolean values
# NumPy converts these from boolean to 1. and 0. later
# yes_no_cols will be re-used for later scoring
yes_no_cols = ["DATA_PLAN"]
df[yes_no_cols] = df[yes_no_cols] == 'yes'


# Pull out fesatures for later scoring
features = df.columns
print(features)

print "~ Feature Variable ~"

# feature variables
X = df.as_matrix().astype(np.float)

print ("\n~~ Input Data Summary ~~\n")
print (df.shape)
print (df.dtypes)


# Importances of features
train_index,test_index = train_test_split(df.index, random_state=4)

print (y[train_index])
print ("\n")

print (X[train_index])
print ("\n")


forest = RF()
forest_fit = forest.fit(X[train_index], y[train_index])


forest_predictions = forest_fit.predict(X[test_index])
print ("\n")

importances = forest_fit.feature_importances_[:10]
print (importances)
print ("\n")
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature Ranking of the Random Forest Churn Prediction Classifier:")

for f in range(10):
    print("%d. %s (%f)" % (f + 1, df.columns[f], importances[indices[f]]))

# Plot the feature importances of the forest
#import pylab as pl
plt.figure()
plt.title("Feature Importance Listed by the Random Forest Churn Prediction Classifier")
plt.bar(range(10), importances[indices], yerr=std[indices], color="r", align="center")
plt.xticks(range(10), indices)
plt.xlim([-1, 10])
plt.show()


print "~ Transforming Data ~"
scaler = StandardScaler()
X = scaler.fit_transform(X)
print "Feature space holds %d observations and %d features" % X.shape
print "Unique target labels:", np.unique(y)


print "~ Building K-Fold Cross-Validations ~"
def run_cv(X, y, clf):
    # construct a K-Fold object
    kf = KFold(n_splits=5, shuffle=True, random_state=4)
    y_pred = y.copy()
    
    # iterate through folds
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf.fit(X_train, y_train)
        y_pred[test_index] = clf.predict(X_test)
    return y_pred


print "~ Evaluating Models ~"    
def accuracy(y_true,y_pred):
    return np.mean(y_true == y_pred)

print "Random Forest Churn Prediction Classifier:"
print "Accuracy = %.3f" % accuracy(y, run_cv(X,y,RF()))


# F1-Scores and Confusion Matrices
def draw_confusion_matrices(confusion_matricies,class_names):
    class_names = class_names.tolist()
    for cm in confusion_matrices:
        classifier, cm = cm[0], cm[1]
        print(cm)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)

        cax = ax.matshow(cm)
        plt.title('Confusion matrix for %s' % classifier)
        fig.colorbar(cax)
        ax.set_xticklabels([''] + class_names)
        ax.set_yticklabels([''] + class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
    
y = np.array(y)
class_names = np.unique(y)


confusion_matrices = {
                  1: {
                    'matrix': confusion_matrix(y,run_cv(X,y,RF())),
                    'title': 'Confusion Matrix of the Random Forest Churn Prediction Classifier',
                   },
}

fix, ax = plt.subplots(figsize=(16, 12))

plt.title('Confusion Matrix of the Classifier')

for ii, values in confusion_matrices.items():
    matrix = values['matrix']
    title = values['title']
    plt.subplot(3, 3, ii) # starts from 1
    plt.title(title);
    sns.heatmap(matrix, annot=True,  fmt='');
    
print  "Random Forest F1 Score" ,f1_score(y,run_cv(X,y,RF()))


# ROC plots
def plot_roc(X, y, clf_class):
    kf = KFold(n_splits=5,shuffle=True, random_state=4)
    y_prob = np.zeros((len(y),2))
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
#    all_tpr = []
    i = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf = clf_class
        clf.fit(X_train,y_train)
        # Predict probabilities, not classes
        y_prob[test_index] = clf.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y[test_index], y_prob[test_index, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
        i = i + 1
    mean_tpr /= kf.get_n_splits(X)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic of the Random Forest Churn Prediction Classifier')
    plt.legend(loc="lower right")
    plt.show()
    
plot_roc(X,y,RF(n_estimators=18))



print "Building K-Fold Cross-Validations with Probabilities"
def run_prob_cv(X, y, clf):
    kf = KFold(n_splits=5,shuffle=True, random_state=4)
    y_prob = np.zeros((len(y),2))
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf.fit(X_train,y_train)
        # Predict probabilities, not classes
        y_prob[test_index] = clf.predict_proba(X_test)
    return y_prob



print "Calculating Calibration and Discrimination"
# Take on RF
pred_prob = run_prob_cv(X, y, RF(n_estimators=10, random_state=4))

# Use 10 estimators so predictions are all multiples of 0.1
pred_churn = pred_prob[:,1].round(1)
is_churn = y == 1

# Number of times a predicted probability is assigned to an observation
counts = pd.value_counts(pred_churn)
counts[:]

print "Calculated Calibration and Discrimination"


# calculate true probabilities
print "~ Calculating the true probabilities ~"
true_prob = {}
for prob in counts.index:
    true_prob[prob] = np.mean(is_churn[pred_churn == prob])
    true_prob = pd.Series(true_prob)

counts = pd.concat([counts,true_prob], axis=1).reset_index()
counts.columns = ['pred_prob', 'count', 'true_prob']
counts.sort_values('pred_prob').reset_index().drop(['index'], axis=1)
print "~ Done calculating the true probabilities ~"


baseline = np.mean(is_churn)
print "~ Done ~"

def calibration(prob,outcome,n_bins=10):
#    """Calibration measurement for a set of predictions.
#    When predicting events at a given probability, how far is frequency
#    of positive outcomes from that probability?
#    NOTE: Lower scores are better
#    prob: array_like, float
#        Probability estimates for a set of events
#    outcome: array_like, bool
#        If event predicted occurred
#    n_bins: int
#        Number of judgement categories to prefrom calculation over.
#        Prediction are binned based on probability, since "descrete" 
#        probabilities aren't required. 
#    """
    prob = np.array(prob)
    outcome = np.array(outcome)

    c = 0.0
    # Construct bins
    judgement_bins = np.arange(n_bins + 1) / n_bins
    # Which bin is each prediction in?
    bin_num = np.digitize(prob,judgement_bins)
    for j_bin in np.unique(bin_num):
        # Is event in bin
        in_bin = bin_num == j_bin
        # Predicted probability taken as average of preds in bin
        predicted_prob = np.mean(prob[in_bin])
        # How often did events in this bin actually happen?
        true_bin_prob = np.mean(outcome[in_bin])
        # Squared distance between predicted and true times num of obs
        c += np.sum(in_bin) * ((predicted_prob - true_bin_prob) ** 2)
    return c / len(prob)

def discrimination(prob,outcome,n_bins=10):
#    """Discrimination measurement for a set of predictions.
#    For each judgement category, how far from the base probability
#    is the true frequency of that bin?
#    NOTE: High scores are better
#    prob: array_like, float
#        Probability estimates for a set of events
#    outcome: array_like, bool
#        If event predicted occurred
#    n_bins: int
#        Number of judgement categories to prefrom calculation over.
#        Prediction are binned based on probability, since "descrete" 
#        probabilities aren't required. 
#    """
    prob = np.array(prob)
    outcome = np.array(outcome)

    d = 0.0
    # Base frequency of outcomes
    base_prob = np.mean(outcome)
    # Construct bins
    judgement_bins = np.arange(n_bins + 1) / n_bins
    # Which bin is each prediction in?
    bin_num = np.digitize(prob,judgement_bins)
    for j_bin in np.unique(bin_num):
        in_bin = bin_num == j_bin
        true_bin_prob = np.mean(outcome[in_bin])
        # Squared distance between true and base times num of obs
        d += np.sum(in_bin) * ((true_bin_prob - base_prob) ** 2)
    return d / len(prob)

def print_measurements(pred_prob):
#    """
#    Print calibration error and discrimination
#    """
    churn_prob, is_churn = pred_prob[:,1], y == 1
    print("  %-20s %.4f" % ("Calibration Error", calibration(churn_prob, is_churn)))
    print("  %-20s %.4f" % ("Discrimination", discrimination(churn_prob,is_churn)))
    print("Note -- Lower calibration is better, higher discrimination is better")


def print_measurements(pred_prob):
    churn_prob, is_churn = pred_prob[:,1], y == 1
    print "  %-20s %.4f" % ("Calibration Error", calibration(churn_prob, is_churn))
    print "  %-20s %.4f" % ("Discrimination", discrimination(churn_prob,is_churn))


print "\nRandom forests:"
print_measurements(run_prob_cv(X,y,RF(n_estimators=18)))

print "\nPlease Note : Lower Calibration Error and Higher Discrimination reflects a better model"


#print "\nK-nearest-neighbors:"
#print_measurements(run_prob_cv(X,y,KNN()))


print '~ Profit Curves ~'
def confusion_rates(cm): 

    tn = cm[0][0]
    fp = cm[0][1] 
    fn = cm[1][0]
    tp = cm[1][1]
    
    N = fp + tn
    P = tp + fn
    
    tpr = tp / P
    fpr = fp / P
    fnr = fn / N
    tnr = tn / N
    
    rates = np.array([[tpr, fpr], [fnr, tnr]])
    
    return rates


def profit_curve(classifier):
    for clf_class in classifier:
        name, clf_class = clf_class[0], clf_class[1]
        clf = clf_class
        fit = clf.fit(X[train_index], y[train_index])
        probabilities = np.array([prob[0] for prob in fit.predict_proba(X[test_index])])
        profit = []
        
        indicies = np.argsort(probabilities)[::1]
    
        for idx in xrange(len(indicies)): 
            pred_true = indicies[:idx]
            ctr = np.arange(indicies.shape[0])
            masked_prediction = np.in1d(ctr, pred_true)
            cm = confusion_matrix(y_test.astype(int), masked_prediction.astype(int))
            rates = confusion_rates(cm)
     
            profit.append(np.sum(np.multiply(rates,cb)))
        
        plt.plot((np.arange(len(indicies)) / len(indicies) * 100), profit, label=name)
    plt.legend(frameon=True, loc="lower left")
    #leg = plt.legend('Random Forest', frameon=True, loc="lower left")
    plt.title("Profits of the Random Forest Churn Prediction Classifier")
    plt.xlabel("Percentage of Test Instances in the Decreasing order by Score)")
    plt.ylabel("Profit")
    plt.show()


# In[30]:

y_test = y[test_index].astype(float)

# Cost-Benefit Matrix
cb = np.array([[4, -5],
               [0, 0]])

# Classifier Effieciency
classifier = [
               ("Random Forest", RF(n_estimators=18))
              ]
               
# Plot profit curves
profit_curve(classifier)

forest = RF(n_estimators=18, random_state=4)
forest_fit = forest.fit(X[train_index], y[train_index])
predictions = forest_fit.predict(X[test_index])

print confusion_matrix(y[test_index],predictions)
print accuracy_score(y[test_index],predictions)
print classification_report(y[test_index],predictions)


# Grid Search and Hyper Parameters
rfc = RF(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True, random_state = 4) 

param_grid = { 
    'n_estimators': [5, 10, 20, 40, 80, 160, 200],
    #'min_samples_leaf': [1, 5, 10, 50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2']
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X[train_index], y[train_index])

means = CV_rfc.cv_results_['mean_test_score']
stds = CV_rfc.cv_results_['std_test_score']

print("Best: %f using %s with %s" % (CV_rfc.best_score_, CV_rfc.best_params_, CV_rfc.best_estimator_))
for params, mean_score, scores in zip(CV_rfc.cv_results_['params'],means,stds):
    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))

forest = CV_rfc.best_estimator_
forest_fit = forest.fit(X[train_index], y[train_index])
predictions = forest_fit.predict(X[test_index])

print confusion_matrix(y[test_index],predictions)
print accuracy_score(y[test_index],predictions)
print classification_report(y[test_index],predictions)

print "Random Forests Senstivity Analysis Train Data:"
plot_roc(X[train_index],y[train_index],CV_rfc.best_estimator_)
print "Random Forests Senstivity Analysis Test Data:"
plot_roc(X[test_index],y[test_index],CV_rfc.best_estimator_)


# Test new data 
df = pd.read_csv('C:/Users/MoideenJ/Desktop/TelcoChurn/Test Data.csv',low_memory=False, header=0)

train_index,test_index = train_test_split(df.index, random_state = 4)
test_df = df.ix[test_index]

# Apply new data to the model
ChurnModel(test_df, CV_rfc.best_estimator_)


# Scoring Model
def ChurnModel(df, clf):
    # Convert yes no columns to bool
    # yes_no_cols already known, stored as a global variable
    df[yes_no_cols] = df[yes_no_cols] == 'yes'
    # features already known, stored as a global variable
    X = df[features].as_matrix().astype(np.float)
    X = scaler.transform(X)
    
    """
    Calculates probability of churn and expected loss, 
    and gathers customer's contact info
    """     
    # Collect customer meta data
    response = df[['NATIONALITY','VF_NUMBER']] 
    
    # Make prediction
    churn_prob = clf.predict_proba(X)
    response['churn_prob'] = churn_prob[:,1]

    print ("\n~~ Export and Print Data ~~\n")

    response.to_csv('C:/Users/MoideenJ/Desktop/TelcoChurn/Input Churn Data/Output.csv', sep=',', encoding='utf-8', index=False)





