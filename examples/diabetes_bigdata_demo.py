from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from BigDataRuleListClassifier import *
from SVMBigDataRuleListClassifier import *
from sklearn.ensemble import RandomForestClassifier
import time

feature_labels = ["#Pregnant","Glucose concentration test","Blood pressure(mmHg)","Triceps skin fold thickness(mm)","2-Hour serum insulin (mu U/ml)","Body mass index","Diabetes pedigree function","Age (years)"]
    
data = fetch_openml("diabetes", parser='auto', version='1') # get dataset
y = (data.target=='tested_positive').astype(int)

###############################################################################

Xtrain, Xtest, ytrain, ytest = train_test_split(data.data, y) # split

t0 = time.time()
# train classifier (allow more iterations for better accuracy)
clf = RuleListClassifier(max_iter=10000, class1label="diabetes", verbose=False)
clf.fit(Xtrain.to_numpy(), ytrain.to_numpy(), feature_labels=feature_labels)
print("RuleListClassifier Accuracy:", clf.score(Xtest, ytest), "Learned interpretable model:\n", clf)
t1 = time.time()

# train classifier (allow more iterations for better accuracy)
bclf = BigDataRuleListClassifier(training_subset=0.1, subset_estimator=RandomForestClassifier(n_estimators=100).fit(Xtrain, ytrain), max_iter=10000, class1label="diabetes", verbose=False)
bclf.fit(Xtrain.values, ytrain.values, feature_labels=feature_labels)
print("BigDataRuleListClassifier Accuracy:", bclf.score(Xtest, ytest), "Learned interpretable model:\n", bclf)
t2 = time.time()

# train classifier (allow more iterations for better accuracy)
sclf = SVMBigDataRuleListClassifier(training_subset=0.1, subsetSVM_C=0.01, max_iter=10000, class1label="diabetes", verbose=False)
sclf.fit(Xtrain.to_numpy(), ytrain.to_numpy(), feature_labels=feature_labels)
print("SVMBigDataRuleListClassifier Accuracy:", bclf.score(Xtest, ytest), "Learned interpretable model:\n", sclf)
t3 = time.time()

print("Comparison\n=========")
print("Time taken for RuleListClassifier: ", t1-t0, "Score achieved:", clf.score(Xtest, ytest))
print("Time taken for BigDataRuleListClassifier: ", t2-t1, "Score achieved:", bclf.score(Xtest, ytest))
print("Time taken for SVMBigDataRuleListClassifier: ", t3-t2, "Score achieved:", sclf.score(Xtest, ytest))
print("========")

###############################################################################

print("RandomForestClassifier Accuracy:", RandomForestClassifier().fit(Xtrain, ytrain).score(Xtest, ytest))