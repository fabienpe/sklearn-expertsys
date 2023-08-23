from RuleListClassifier import *
import sklearn.ensemble
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

dataseturls = ["https://archive.ics.uci.edu/ml/datasets/Iris", "https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes"]
datasets = ["iris", "diabetes"]
data_feature_labels = [
    ["Sepal length", "Sepal width", "Petal length", "Petal width"],
    ["#Pregnant","Glucose concentration demo","Blood pressure(mmHg)","Triceps skin fold thickness(mm)","2-Hour serum insulin (mu U/ml)","Body mass index","Diabetes pedigree function","Age (years)"]
]
data_class1_labels = ["Iris-versicolor", "tested_positive"]
for i in range(0, len(datasets)):
    print('--------')
    print(f'DATASET: {datasets[i]} ({dataseturls[i]})')
    data = fetch_openml(datasets[i], parser='auto', version='1')
    y = (data.target==data_class1_labels[i]).astype(int)

    Xtrain, Xtest, ytrain, ytest = train_test_split(data.data, y) 
    
    clf = RuleListClassifier(max_iter=50000, n_chains=3, class1label=data_class1_labels[i], verbose=False)
    clf.fit(Xtrain.to_numpy(), ytrain.to_numpy(), feature_labels=data_feature_labels[i])
    
    print(f'accuracy: {clf.score(Xtest, ytest)}')
    print(f'rules:\n{clf}')
    print(f'Random Forest accuracy: {sklearn.ensemble.RandomForestClassifier().fit(Xtrain, ytrain).score}\n({Xtest}, {ytest})')