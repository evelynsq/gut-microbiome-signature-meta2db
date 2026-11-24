"""
Script that should be the starting-off point for building deep-learning model.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skbio.stats.composition import clr
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from combatlearn import ComBat


##### read in data #####
combined_df = pd.read_parquet('meta2db_combined_fecal_abundances.parquet') # I recommend using this for faster reading
# combined_df = pd.read_csv('meta2db_combined_fecal_abundances.csv', index_col=0)
print("Finished reading in abundance file...")
print(f"Shape of dataframe: {combined_df.shape}")
print(combined_df.head())

# get rid of pre-diseased samples to simplify analysis
combined_df = combined_df[combined_df['health_disease_stat'].isin(['control', 'diseased'])]


##### focus on only the 'neurological' disease category #####
print(combined_df['disease_category'].value_counts()) # see how many samples per each disease category
disease_cat_focus = 'neurological'
disease_rows = combined_df['disease_category'].isin([disease_cat_focus])
neurological_df = combined_df.loc[disease_rows, :]

# exclude the 2017 study, very small sample size which will cause problems for batch correction
neurological_df = neurological_df.loc[neurological_df['project_name'] != '2017_Bedarf_PD'].copy()


##### separate into training and outcome data #####
drop_categorical_cols = ['health_disease_stat', 'host_age', 'host_sex', 
                         'project_name', 'disease_category']
X = neurological_df.drop(columns=drop_categorical_cols).copy()
y = neurological_df['health_disease_stat']
print(f"{X.shape[1]} genus-level features")

# encode disease status as 0: Control and 1: diseased
y = y.map({'control': 0, 'diseased': 1}).astype(int)
print(y.value_counts())


##### CLR transform abundance data first #####
pseudo_count = 0.5
clr_abundances = clr(X + pseudo_count)
clr_df = pd.DataFrame(clr_abundances, columns=X.columns, index=X.index)
print("CLR transformed data...")


#### train/test split data #####
strat_disease_study = neurological_df['project_name'] + "_" + y.astype(str)
XTrain, XTest, ytrain, ytest = train_test_split(clr_df, y, test_size=0.2,
                                                random_state=42,
                                                stratify=strat_disease_study) # stratify by disease status + study
print("Data has been split into 80/20 train vs. test...")


##### Batch Correction with combatlearn #####
"""
Attempted to also include sex and age as covariates, but a large percentage of
those samples have no sex or age information available in metadata.
"""
# filter out non-zero variance features in the training data - might throw error with ComBat
nonzero_var_cols = XTrain.columns[XTrain.var() > 1e-8]
XTrain = XTrain[nonzero_var_cols]
XTest = XTest[nonzero_var_cols]

# correct for different study batch effect
combat = ComBat(batch = neurological_df['project_name'], method='johnson')

XTrain_combat = combat.fit_transform(XTrain)
XTest_combat = combat.transform(XTest)
print("Corrected for study batch effect with combatlearn...")


##### scale data #####
scaler = StandardScaler()
XSTrain = scaler.fit_transform(XTrain_combat)
XSTest = scaler.transform(XTest_combat)
print("Features have been standardized...")


mlp_code = """
# if you are curious, this is how a baseline MLP performs
print("Running a baseline MLP...")
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    alpha=0.1,
    max_iter=500,
    early_stopping=True,
    random_state=42)

mlp.fit(XSTrain, ytrain)
predY = mlp.predict(XSTest)
print(f"MLP Test Accuracy: {accuracy_score(ytest, predY):.4f}")

predY_prob = mlp.predict_proba(XSTest)[:, 1]
auc = roc_auc_score(ytest, predY_prob)
print(f"MLP ROC AUC: {auc:.4f}")
"""

###### copy this file, change the filename and make        ######
###### your deep learning model implementation below here  ######