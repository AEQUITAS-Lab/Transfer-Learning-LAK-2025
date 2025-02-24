import os
import argparse
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Model features
cols = ['units', 'units_failed', 'units_incompleted', 'units_withdrawn', 'gpa_cumulative', 'age', 'gpa_avg', 
        'gpa_stddev', 'gpa_zscore_avg', 'gpa_zscore_stddev', 'STDNT_FEMALE', 'STDNT_ASIAN_IND', 
        'STDNT_BLACK_IND', 'STDNT_HSPNC_IND', 'STDNT_NTV_AMRCN_HWIAN_IND', 'STDNT_ETHNC_NAN', 'urm_status', 'cip2_major_1']

# Models
class RetentionFeatureExtractor(nn.Module):
    def __init__(self, input_dim):
        super(RetentionFeatureExtractor, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.layer1(x)))
        return x

class RetentionBottleneck(nn.Module):
    def __init__(self):
        super(RetentionBottleneck, self).__init__()
        self.layer2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.5)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.leaky_relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)
        return x

class RetentionClassifier(nn.Module):
    def __init__(self):
        super(RetentionClassifier, self).__init__()
        self.layer3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.5)
        self.leaky_relu = nn.LeakyReLU()
        self.layer4 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.bn3(self.layer3(x)))
        x = self.dropout(x)
        x = self.sigmoid(self.layer4(x))
        return x

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

def clean(data):
    data['STDNT_FEMALE'] = data['sex'].apply(lambda x: 1 if x == 'Female' else 0)
    data['STDNT_ASIAN_IND'] = data['ethnicity'].apply(lambda x: 1 if x == 'Asian' else 0)
    data['STDNT_BLACK_IND'] = data['ethnicity'].apply(lambda x: 1 if x == 'Black' else 0)
    data['STDNT_HSPNC_IND'] = data['ethnicity'].apply(lambda x: 1 if x == 'Hispanic' else 0)
    data['STDNT_NTV_AMRCN_HWIAN_IND'] = data['ethnicity'].apply(lambda x: 1 if x in ['Native Amr', 'Hawaiian'] else 0)
    data['STDNT_ETHNC_NAN'] = data['ethnicity'].apply(lambda x: 1 if x == 'Not Indic' else 0)
    data['urm_status'] = data['urm_status'].apply(lambda x: 1 if x == 'Underrepresented Minority' else 0)
    data['retention'] = data['retention'].astype(int)
    return data

def train_neural_network_model(train_features, train_labels, model_save_path, model_suffix, base_name):
    # Initialize and train neural network model
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    train_tensor = TensorDataset(torch.tensor(train_features, dtype=torch.float32), 
                                 torch.tensor(train_labels.values, dtype=torch.float32).view(-1, 1))
    
    netF = RetentionFeatureExtractor(train_features.shape[1])
    netB = RetentionBottleneck()
    netC = RetentionClassifier()

    criterion = nn.BCELoss()
    param_group = list(netF.parameters()) + list(netB.parameters()) + list(netC.parameters())
    optimizer = torch.optim.Adam(param_group, lr=0.001)

    train_loader = DataLoader(train_tensor, batch_size=32, shuffle=True)
    netF.train()
    netB.train()
    netC.train()

    for epoch in range(10):
        for features, labels in train_loader:
            if features.size(0) <= 1:
                continue

            optimizer.zero_grad()
            outputs = netF(features)
            outputs = netB(outputs)
            outputs = netC(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    os.makedirs(model_save_path, exist_ok=True)
    torch.save(netF.state_dict(), os.path.join(model_save_path, f"retention_netF_{base_name}_{model_suffix}.pth"))
    torch.save(netB.state_dict(), os.path.join(model_save_path, f"retention_netB_{base_name}_{model_suffix}.pth"))
    torch.save(netC.state_dict(), os.path.join(model_save_path, f"retention_netC_{base_name}_{model_suffix}.pth"))
    joblib.dump(scaler, os.path.join(model_save_path, f"scaler_{base_name}_{model_suffix}.pkl"))

def train_logistic_regression_model(train_features, train_labels, model_save_path, model_suffix, base_name):
    # Initialize and train logistic regression model
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    train_tensor = TensorDataset(torch.tensor(train_features, dtype=torch.float32), 
                                 torch.tensor(train_labels.values, dtype=torch.float32).view(-1, 1))

    logistic_model = LogisticRegressionModel(train_features.shape[1])

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(logistic_model.parameters(), lr=0.001, weight_decay=0.01)

    train_loader = DataLoader(train_tensor, batch_size=32, shuffle=True)
    logistic_model.train()

    for epoch in range(10):
        for features, labels in train_loader:
            if features.size(0) <= 1:
                continue

            optimizer.zero_grad()
            outputs = logistic_model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    os.makedirs(model_save_path, exist_ok=True)
    torch.save(logistic_model.state_dict(), os.path.join(model_save_path, f"logistic_model_{base_name}_{model_suffix}.pth"))
    joblib.dump(scaler, os.path.join(model_save_path, f"logistic_scaler_{base_name}_{model_suffix}.pkl"))

def main(file_path, models):
    # Extract the base name of the input file
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Load and clean data
    src = pd.read_feather(file_path)
    src = clean(src)
    src = src.dropna(subset=['gpa_cumulative', 'age', 'gpa_avg', 'gpa_stddev', 'gpa_zscore_avg', 'gpa_zscore_stddev'])
    src['cip2_major_1'] = src['cip2_major_1'].fillna(0).replace('MISSING', 0).astype(str)

    # Prepare original and SMOTE datasets
    train_data_raw = src[(src['year'] >= 2012) & (src['year'] <= 2018)]
    train_features_raw = train_data_raw[cols]
    train_labels_raw = train_data_raw['retention']

    smote = SMOTE(sampling_strategy=1, random_state=42)
    train_features_smote, train_labels_smote = smote.fit_resample(train_features_raw, train_labels_raw)

    # Train models as specified
    if 'nn_raw' in models:
        train_neural_network_model(train_features_raw, train_labels_raw, 'trained_model_test', 'raw', base_name)
    if 'nn_smote' in models:
        train_neural_network_model(train_features_smote, train_labels_smote, 'trained_model_test', 'smote', base_name)
    if 'logistic_raw' in models:
        train_logistic_regression_model(train_features_raw, train_labels_raw, 'logistic_model_test', 'raw', base_name)
    if 'logistic_smote' in models:
        train_logistic_regression_model(train_features_smote, train_labels_smote, 'logistic_model_test', 'smote', base_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models on data.")
    parser.add_argument("file_path", type=str, help="Path to the Feather file")
    parser.add_argument("--models", nargs='+', choices=['nn_raw', 'nn_smote', 'logistic_raw', 'logistic_smote'],
                        help="Models to train: 'nn_raw', 'nn_smote', 'logistic_raw', 'logistic_smote'", required=True)
    args = parser.parse_args()
    main(args.file_path, args.models)
