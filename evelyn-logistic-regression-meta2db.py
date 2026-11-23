import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from skbio.stats.composition import clr

# read in data
combined_df = pd.read_parquet('meta2db_combined_fecal_abundances.parquet') # I recommend using this for faster reading
# combined_df = pd.read_csv('meta2db_combined_fecal_abundances.csv', index_col=0)
print("Finished reading in abundance file...")
print(combined_df.head())

# get rid of pre-diseased samples to simplify analysis
combined_df = combined_df[combined_df['health_disease_stat'].isin(['control', 'diseased'])]

# separate into training and outcome data
X = combined_df.drop(columns=['project_name', 'disease_category', 'health_disease_stat'])
y = combined_df['health_disease_stat']

# encode disease status as 0: Control and 1: diseased
y = y.map({'control': 0, 'diseased': 1}).astype(int)
print(y.value_counts())

# CLR transform abundance data first
# need to add a pseudocount to deal with zeros - 1 for raw counts
pseudo_count = 1

# apply the CLR transformation
clr_abundances = clr(X + pseudo_count)
clr_df = pd.DataFrame(clr_abundances, columns=X.columns, index=X.index)
print("CLR transformed data...")

# train/test split data
XTrain, XTest, ytrain, ytest = train_test_split(clr_abundances, y, test_size=0.2, 
                                                random_state=42, stratify=y)
print("Data has been split into 80/20 train vs. test...")

# scale data
scaler = StandardScaler()
XSTrain = scaler.fit_transform(XTrain)
XSTest = scaler.transform(XTest)
print("Data has been scaled...")

# baseline model
print("Running a baseline logistic regression with elastic net...")
base_model = LogisticRegression(penalty='l1', solver='liblinear', C=0.5, 
                                class_weight='balanced').fit(XSTrain, ytrain)
predY = base_model.predict(XSTest)

print(f"Accuracy of baseline model: {accuracy_score(ytest, predY)}")

# still testing out this part
print("Trying out grid search to find best parameters...")
lr_clf = LogisticRegression(class_weight='balanced')

# try grid search to test for different parameters
param_grid_lr = {
    "l1_ratio": np.linspace(0, 1.0, 10),
    "C": [0.001, 0.03, 0.1, 0.25, 0.5, 1, 3, 10],
    "penalty": ['elasticnet'],
    "solver": ['saga'],
    "max_iter": [2000]
}

# leave out 20% of data per fold
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# do a grid search with the combinations of hyperparameters above
random_search = RandomizedSearchCV(
    estimator=lr_clf,
    param_distributions=param_grid_lr,
    cv=kfold,
    scoring='roc_auc'
)

# run the actual search
random_search.fit(XSTrain, ytrain)

print(f"Best ROC AUC: {random_search.best_score_:.3f}")
print("Best Hyperparameters:")
print(random_search.best_params_)

# get the best model
best_model = random_search.best_estimator_

predY = best_model.predict(XSTest)
print(f"Test accuracy of best model: {accuracy_score(ytest, predY)}")