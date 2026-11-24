import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from skbio.stats.composition import clr


# read in data - fix your path as needed
combined_df = pd.read_parquet('meta2db_combined_fecal_abundances.parquet') # I recommend using this for faster reading
# combined_df = pd.read_csv('meta2db_combined_fecal_abundances.csv', index_col=0)
print("Finished reading in abundance file...")
print(f"Shape of dataframe: {combined_df.shape}")
print(combined_df.head())

# get rid of pre-diseased samples to simplify analysis
combined_df = combined_df[combined_df['health_disease_stat'].isin(['control', 'diseased'])]

# try out only a certain disease category
print(combined_df['disease_category'].value_counts()) # see how many samples per each disease category
disease_cat_focus = 'age-related macular degeneration'
disease_rows = combined_df['disease_category'].isin([disease_cat_focus, 'healthy subject study']) # don't filter out control data
combined_df = combined_df.loc[disease_rows, :]

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
print("Features have been standardized...")


### Try out Logistic Regression baseline classifier
logreg_code = """print("Running a baseline logistic regression with elastic net...")
lr = LogisticRegression(penalty='l1', solver='liblinear', C=0.5, 
                        class_weight='balanced').fit(XSTrain, ytrain)
predY = lr.predict(XSTest)
predY_prob = lr.predict_proba(XSTest)[:, 1]
print(f"Logistic Regression Test Accuracy: {accuracy_score(ytest, predY):.4f}")

auc = roc_auc_score(ytest, predY_prob)
print(f"Logistic Regression ROC AUC: {auc:.4f}")"""


### Try out MLP classifier
print("Running a baseline MLP...")
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    alpha=0.005,
    max_iter=500,
    early_stopping=True,
    random_state=42)

mlp.fit(XSTrain, ytrain)
predY = mlp.predict(XSTest)
print(f"MLP Test Accuracy: {accuracy_score(ytest, predY):.4f}")

predY_prob = mlp.predict_proba(XSTest)[:, 1]
auc = roc_auc_score(ytest, predY_prob)
print(f"MLP ROC AUC: {auc:.4f}")