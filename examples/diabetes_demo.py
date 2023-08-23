from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from RuleListClassifier import *
from sklearn.ensemble import RandomForestClassifier

feature_labels = ["#Pregnant","Glucose concentration test","Blood pressure(mmHg)","Triceps skin fold thickness(mm)","2-Hour serum insulin (mu U/ml)","Body mass index","Diabetes pedigree function","Age (years)"]
    
data = fetch_openml("diabetes", parser='auto', version='1') # get dataset
#y = -(data.target-1)/2 # target labels (0: healthy, or 1: diabetes) - the original dataset contains -1 for diabetes and +1 for healthy
y = (data.target=='tested_positive').astype(int)

###############################################################################

Xtrain, Xtest, ytrain, ytest = train_test_split(data.data, y) # split

# train classifier (allow more iterations for better accuracy)
clf = RuleListClassifier(max_iter=10000, class1label="diabetes", verbose=False)
clf.fit(Xtrain.to_numpy(), ytrain.to_numpy(), feature_labels=feature_labels)

print("RuleListClassifier Accuracy:", clf.score(Xtest, ytest), "Learned interpretable model:\n", clf)

###############################################################################

print("RandomForestClassifier Accuracy:", RandomForestClassifier().fit(Xtrain, ytrain).score(Xtest, ytest))