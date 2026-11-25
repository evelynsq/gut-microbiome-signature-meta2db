"""
Script that builds a supervised denoising autoencoder for pulmonary disease studies 
specifically. (disease vs. healthy)

I was checking out different disease type comparisons, and a baseline MLP for 
pulmonary diseases was already at 82% which looked more promising than the neurological
MLP at 66%.

And looking into it further, pulmonary disease classification actually can perform 
better with using the supervised autoencoder model :)

I will look more into ways to improve accuracy, but the current implementation right now has:
seed = 42
test accuracy: 86%
test AUC: 0.923

seed = 0
test accuracy: 92%
test AUC: 0.981
"""

# preprocessing data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, f1_score
from sklearn.neural_network import MLPClassifier
from skbio.stats.composition import clr # CLR transformation
from combatlearn import ComBat # study batch effect correction

# building autoencoder model
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim

print("Finished importing libraries")


##### read in data #####
combined_df = pd.read_parquet('meta2db_combined_fecal_abundances.parquet') # I recommend using this for faster reading
# combined_df = pd.read_csv('meta2db_combined_fecal_abundances.csv', index_col=0)
print("Finished reading in abundance file...")
print(f"Shape of dataframe: {combined_df.shape}")
print(combined_df.head(), "\n")

# get rid of pre-diseased samples to simplify analysis
combined_df = combined_df[combined_df['health_disease_stat'].isin(['control', 'diseased'])]


##### focus on only the 'pulmonary' disease category studies #####
print(combined_df['disease_category'].value_counts(), "\n") # see how many samples per each disease category
disease_cat_focus = 'pulmonary'
disease_rows = combined_df['disease_category'].isin([disease_cat_focus])
pulmonary_df = combined_df.loc[disease_rows, :]
print(pulmonary_df.groupby("project_name")["health_disease_stat"].value_counts(), "\n")


##### separate into training and outcome data #####
drop_categorical_cols = ['health_disease_stat', 'host_age', 'host_sex', 
                         'project_name', 'disease_category']
X = pulmonary_df.drop(columns=drop_categorical_cols).copy()
y = pulmonary_df['health_disease_stat']
print(f"{X.shape[1]} genus-level features")

# encode disease status as 0: Control and 1: diseased
y = y.map({'control': 0, 'diseased': 1}).astype(int)
print(y.value_counts(), "\n")


##### CLR transform microbiome abundance data first #####
pseudo_count = 0.5
clr_abundances = clr(X + pseudo_count)
clr_df = pd.DataFrame(clr_abundances, columns=X.columns, index=X.index)
print("CLR transformed data...")


#### Train/test split data #####
strat_disease_study = pulmonary_df['project_name'] + "_" + y.astype(str)
XTrain, XTest, ytrain, ytest = train_test_split(clr_df, y, test_size=0.2,
                                                random_state=42,
                                                stratify=strat_disease_study) # stratify by disease status + study
print("Data has been split into 80/20 train vs. test...")

##### Batch Correction with combatlearn #####
# Previously attempted to also include sex and age as covariates, but a large percentage of
# those samples have no sex or age information available in metadata to use.

# filter out non-zero variance features in the training data - might throw error with ComBat
nonzero_var_cols = XTrain.columns[XTrain.var() > 1e-8]
XTrain = XTrain[nonzero_var_cols]
XTest = XTest[nonzero_var_cols]

# correct for different study batch effect
combat = ComBat(batch = pulmonary_df['project_name'], method='johnson')
XTrain_combat = combat.fit_transform(XTrain)
XTest_combat = combat.transform(XTest)
print("Corrected for study batch effect with combatlearn...")


##### scale data #####
scaler = StandardScaler()
XSTrain = scaler.fit_transform(XTrain_combat)
XSTest = scaler.transform(XTest_combat)
print("Features have been standardized...")


##### Baseline MLP for comparison #####
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
print(f"MLP Test Accuracy: {accuracy_score(ytest, predY)}")

predY_prob = mlp.predict_proba(XSTest)[:, 1]
auc = roc_auc_score(ytest, predY_prob)
print(f"MLP ROC AUC: {auc}")


###### Supervised Denoising Autoencoder Implementation ######

# References for learning how to build a denoising autoencoder: 
# https://www.geeksforgeeks.org/machine-learning/denoising-autoencoders-in-machine-learning/
# https://github.com/mortezamg63/Supervised-autoencoder/blob/main/SAE_SUSY.py
# https://www.geeksforgeeks.org/deep-learning/batch-normalization-implementation-in-pytorch/

# set up device for use based on what is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
num_epochs = 100

# weights for classification and reconstruction loss
recon_weight = 0.1
ce_weight = 1

noise_std = 0.15 # determines the level of noise to add on input features
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
k = 5 # for stratified kfold validation split

class SupervisedAutoencoder(nn.Module):
    def __init__(self, input_features, hidden_dim1, latent_dim):
        super().__init__()
        # encoder layers
        self.fc_enc1 = nn.Linear(input_features, hidden_dim1)
        self.dropout_enc = nn.Dropout(p=0.5)
        self.batch_norm_enc = nn.BatchNorm1d(hidden_dim1)
        self.fc_enc2 = nn.Linear(hidden_dim1, latent_dim)

        # decoder layers
        self.fc_dec1 = nn.Linear(latent_dim, hidden_dim1)
        self.batch_norm_dec = nn.BatchNorm1d(hidden_dim1)
        self.fc_dec2 = nn.Linear(hidden_dim1, input_features)

        # activation layers
        self.lrelu = nn.LeakyReLU()

        # disease status classification layer: gives probs for each of the classes
        self.classifier = nn.Linear(latent_dim, 2)


    def forward(self, x):
        # encoding: linear -> batch norm -> relu -> dropout -> linear -> relu
        enc1 = self.batch_norm_enc(self.fc_enc1(x))
        enc1 = self.dropout_enc(self.lrelu(enc1))
        enc2 = self.lrelu(self.fc_enc2(enc1))

        # decoding: linear -> batch norm -> relu -> reconstructed input
        dec1 = self.batch_norm_dec(self.fc_dec1(enc2))
        dec1 = self.lrelu(dec1)
        recon_x = self.fc_dec2(dec1) # the reconstructed input

        # classification: get the logits
        cl1_logits = self.classifier(enc2)

        return cl1_logits, recon_x
    

# split 80/20 train vs. test
strat_disease_study = pulmonary_df["project_name"] + "_" + y.astype(str)
XTrain, XTest, ytrain, ytest = train_test_split(clr_df, y, test_size=0.2, random_state=seed,
    stratify=strat_disease_study)
print(f"Split by train/test: {len(XTrain)} train samples, {len(XTest)} test samples")


def preprocess_data(Xtrain_data, Xeval_data):
    """
    Fit + transform variance filtering, ComBat batch correction, and 
    scale features on training data and apply transformations to evaluation data.
    """
    # filter out the features with super small variance
    nonzero_var_cols = Xtrain_data.columns[Xtrain_data.var() > 1e-8]
    Xtrain_filt = Xtrain_data[nonzero_var_cols]
    Xeval_filt = Xeval_data[nonzero_var_cols]

    # account for the study batch effect
    combat = ComBat(batch=pulmonary_df['project_name'], method="johnson")
    Xtrain_batch_corr = combat.fit_transform(Xtrain_filt)
    Xeval_batch_corr = combat.transform(Xeval_filt)

    # standardize data
    scaler = StandardScaler()
    XSTrain = scaler.fit_transform(Xtrain_batch_corr)
    XSVal = scaler.transform(Xeval_batch_corr)

    return XSTrain, XSVal


def train_val(XSTrain, ytrain, XSVal):
    """
    Train a model per each fold and return the predicted probabilities for 
    validation data.
    """
    # convert the data into tensors
    XTrain_ten = torch.tensor(XSTrain, dtype=torch.float32)
    ytrain_ten = torch.tensor(ytrain.to_numpy(), dtype=torch.long)
    XVal_ten = torch.tensor(XSVal, dtype=torch.float32).to(device)

    train_loader = DataLoader(TensorDataset(XTrain_ten, ytrain_ten), 
                              batch_size=batch_size, shuffle=True)

    # autoencoder model
    model = SupervisedAutoencoder(input_features=XTrain_ten.shape[1], 
                                  hidden_dim1=128, latent_dim=48).to(device)

    # optimizer and learning rate scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # loss
    mse_loss = nn.MSELoss()
    cross_entropy_loss = nn.CrossEntropyLoss()

    final_train_loss = 0
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            # clear out old gradients for each training step
            optimizer.zero_grad()

            # add Gaussian noise to the training data features
            noisy_features = features + noise_std * torch.randn_like(features)

            # forward pass of noisy data
            logits, recon_features = model(noisy_features)

            # check loss
            ce_loss = cross_entropy_loss(logits, labels) # logits
            recon_loss = mse_loss(recon_features, features) # reconstruction loss
            loss = (ce_loss * ce_weight) + (recon_loss * recon_weight) # overall loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(features)

        scheduler.step()
        final_train_loss = train_loss / len(train_loader.dataset)

    model.eval()
    # make predictions on test data and assess accuracy
    with torch.no_grad():
        val_logits, _ = model(XVal_ten)

        # convert to probabilities
        probs = torch.softmax(val_logits, dim=1).cpu().numpy()

    return probs, final_train_loss


##### do StratifiedKFold validation on the training/validation data #####

print("Performing stratified k-fold cross validation...")
# make sure we keep the same proportion of diseased patients per project in each fold
strat_disease_study = pulmonary_df.loc[XTrain.index, "project_name"] + "_" + ytrain.astype(str)
skf = StratifiedKFold(k, shuffle=True, random_state=seed)

validation_accur = [None]*k
validation_aucs = [None]*k

# be sure to also stratify while keeping same disease + study proportions
for i, (train_index, val_index) in enumerate(skf.split(XTrain, strat_disease_study)):

    # get the train and validation data of the current fold
    ytrain_split = ytrain.iloc[train_index]
    yval_split = ytrain.iloc[val_index]

    # preprocess + split the current fold data, then train on the processed data
    XSTrain_split, XSVal_split = preprocess_data(XTrain.iloc[train_index,:], 
                                                 XTrain.iloc[val_index,:])
    val_probs, train_loss = train_val(XSTrain_split, ytrain_split, XSVal_split)

    # get predictions from the probabilities returned - take the label with highest prob
    predY = np.argmax(val_probs, axis=1)
    prob_diseased = val_probs[:, 1]

    # get accuracy and AUC of current prediction
    accur = accuracy_score(yval_split, predY)
    auc = roc_auc_score(yval_split, prob_diseased)

    validation_accur[i] = accur
    validation_aucs[i] = auc

    print(f"Fold {i+1} - Final train loss: {train_loss:.5f}, Validation accuracy: {accur:.5f}, Validation AUC: {auc:.5f}")

print(f"\nFinal validation accuracies for each fold:\n {validation_accur}")
print(f"Final validation AUCs for each fold:\n {validation_aucs}\n")


##### Evaluate on the test data now (was held out previously) #####
# preprocess the train and test data
XSTrain, XSTest = preprocess_data(XTrain, XTest)

# get probabilities and final loss from training and test evaluation
probs, final_train_loss = train_val(XSTrain, ytrain, XSTest)
prob_diseased = probs[:, 1]
predY = np.argmax(probs, axis=1)

test_accur = accuracy_score(ytest, predY)
test_auc = roc_auc_score(ytest, prob_diseased)
print(f"Test accuracy: {test_accur}, Test AUC: {test_auc}")


##### Plot results #####
# confusion matrix
cm = confusion_matrix(ytest, predY)
ConfusionMatrixDisplay(cm, display_labels=['healthy', 'diseased']).plot(cmap='Blues')
plt.title("Confusion Matrix of Test Set Predictions")
plt.savefig('images/confusion-matrix-test-set.png', dpi=300)

# ROC AUC curve
RocCurveDisplay.from_predictions(ytest, prob_diseased)
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier", alpha=0.5, color='gray')
plt.title("ROC AUC Curve of Test Set Predictions")
plt.legend()
plt.savefig('images/roc-curve-test-set.png', dpi=300)

print("Test set metric plots have been saved to the 'images' subdirectory.")