import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight


def preprocess_data(data):
    """
    preprocess_data: Preprocess the data for training and evaluation.
    arg data: DataFrame containing the data to be preprocessed.
    return: DataFrame containing the preprocessed data and target labels.
    """
    data = data.drop(columns=[
        'RedFighter', 'BlueFighter', 'Date', 'Location', 'Country',
        'TitleBout', 'FinishDetails', 'Finish', 'TotalFightTimeSecs',
        'FinishRoundTime'], errors='ignore')

    numerical_columns = [
        'RedOdds', 'BlueOdds', 'RedExpectedValue', 'BlueExpectedValue', 
        'BlueAvgSigStrLanded', 'BlueAvgSigStrPct', 'BlueAvgSubAtt', 'BlueAvgTDLanded', 
        'BlueAvgTDPct', 'BlueLongestWinStreak', 'BlueLosses', 'BlueTotalRoundsFought', 
        'BlueTotalTitleBouts', 'BlueWinsByDecisionMajority', 'BlueWinsByDecisionSplit', 
        'BlueWinsByDecisionUnanimous', 'BlueWinsByKO', 'BlueWinsBySubmission', 
        'BlueWinsByTKODoctorStoppage', 'BlueWins', 'BlueHeightCms', 'BlueReachCms', 
        'BlueWeightLbs', 'RedCurrentLoseStreak', 'RedCurrentWinStreak', 'RedDraws', 
        'RedAvgSigStrLanded', 'RedAvgSigStrPct', 'RedAvgSubAtt', 'RedAvgTDLanded', 
        'RedAvgTDPct', 'RedLongestWinStreak', 'RedLosses', 'RedTotalRoundsFought', 
        'RedTotalTitleBouts', 'RedWinsByDecisionMajority', 'RedWinsByDecisionSplit', 
        'RedWinsByDecisionUnanimous', 'RedWinsByKO', 'RedWinsBySubmission', 
        'RedWinsByTKODoctorStoppage', 'RedWins', 'RedHeightCms', 'RedReachCms', 
        'RedWeightLbs', 'RedAge', 'BlueAge', 'LoseStreakDif', 'WinStreakDif', 
        'LongestWinStreakDif', 'WinDif', 'LossDif', 'TotalRoundDif', 'TotalTitleBoutDif', 
        'KODif', 'SubDif', 'HeightDif', 'ReachDif', 'AgeDif', 'SigStrDif', 'AvgSubAttDif', 
        'AvgTDDif', 'BMatchWCRank', 'RMatchWCRank', 'RWFlyweightRank', 'RWFeatherweightRank', 
        'RWStrawweightRank', 'RWBantamweightRank', 'RHeavyweightRank', 'RLightHeavyweightRank', 
        'RMiddleweightRank', 'RWelterweightRank', 'RLightweightRank', 'RFeatherweightRank', 
        'RBantamweightRank', 'RFlyweightRank', 'RPFPRank', 'BWFlyweightRank', 'BWFeatherweightRank', 
        'BWStrawweightRank', 'BWBantamweightRank', 'BHeavyweightRank', 'BLightHeavyweightRank', 
        'BMiddleweightRank', 'BWelterweightRank', 'BLightweightRank', 'BFeatherweightRank', 
        'BBantamweightRank', 'BFlyweightRank', 'BPFPRank', 'FinishRound',
        'RedDecOdds', 'BlueDecOdds', 'RSubOdds', 'BSubOdds', 'RKOOdds', 'BKOOdds'
    ]

    categorical_columns = [
        'WeightClass', 'Gender', 'BlueStance', 'RedStance', 'BetterRank'
    ]

    # ['RedFighter', 'BlueFighter', 'Date', 'Location', 'Country', 
    #     'TitleBout']

    y = data['Winner'] if 'Winner' in data.columns else None
    X = data.drop(columns=['Winner']) if 'Winner' in data.columns else data.copy()

    for col in numerical_columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X[numerical_columns] = X[numerical_columns].fillna(0)

    scaler = MinMaxScaler()
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

    for col in categorical_columns:
        X[col] = X[col].fillna('Unknown')
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    return X, y
    # label_encoders = {}
    # for col in categorical_columns:
    #     X[col] = X[col].fillna('Unknown')
    #     le = LabelEncoder()
    #     X[col] = le.fit_transform(X[col].astype(str))
    #     label_encoders[col] = le
    #return X, y, scaler, le

def preprocess_input_train(df, numerical_columns, categorical_columns):
    X = df.copy()
    label_encoders = {}

    for col in numerical_columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

    scaler = MinMaxScaler()
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

    for col in categorical_columns:
        X[col] = X[col].fillna("Unknown").astype(str)
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    return X, scaler, label_encoders

def preprocess_input_test(df, scaler, label_encoders, numerical_columns, categorical_columns):
    X = df.copy()

    for col in numerical_columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    X[numerical_columns] = scaler.transform(X[numerical_columns])

    for col in categorical_columns:
        X[col] = X[col].fillna("Unknown").astype(str)
        le = label_encoders[col]
        X[col] = X[col].apply(lambda x: x if x in le.classes_ else "Unknown")
        if "Unknown" not in le.classes_:
            le.classes_ = np.append(le.classes_, "Unknown")
        X[col] = le.transform(X[col])

    return X   

def tune_models(X_train, y_train):

    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_resampled)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]} 

    clf1 = RandomForestClassifier(class_weight='balanced', random_state=42)
    clf2 = xgb.XGBClassifier(scale_pos_weight=10, random_state=42, use_label_encoder=False, eval_metric='logloss')
    clf3 = LogisticRegression(class_weight='balanced', random_state=42)


    rf_param_grid = {
    'n_estimators': [100, 200, 300, 500],  
    'max_depth': [None, 10, 20, 30],  
    'min_samples_split': [2, 5, 10],  
    'min_samples_leaf': [1, 2, 4],  
    'max_features': ['sqrt', 'log2'], 
    'class_weight': ['balanced', None]  
    }

    xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'scale_pos_weight': [1, 5, 10]
    }

    lr_param_grid = {
    'C': [0.01, 0.1, 1.0, 10, 100],  
    'penalty': ['l2'],  
    'solver': ['liblinear', 'saga'], 
    'class_weight': ['balanced', None]  
    }

    rf_search = GridSearchCV(clf1, rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    xgb_search = GridSearchCV(clf2, xgb_param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    lr_search = GridSearchCV(clf3, lr_param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

    rf_search.fit(X_resampled, y_resampled)
    xgb_search.fit(X_resampled, y_resampled)
    lr_search.fit(X_resampled, y_resampled)

    best_rf = rf_search.best_estimator_
    best_xgb = xgb_search.best_estimator_
    best_lr = lr_search.best_estimator_

    ensemble_model = VotingClassifier(estimators=[('rf', best_rf), ('xgb', best_xgb), ('lr', best_lr)], voting='soft')

    ensemble_model.fit(X_resampled, y_resampled)

    return ensemble_model

def train_model(X_train, y_train):
    """
    train_model: Train the ensemble model without any hyperparameter tuning.
    """
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    clf1 = RandomForestClassifier(random_state=42)
    clf2 = xgb.XGBClassifier(scale_pos_weight=10, random_state=42, use_label_encoder=False, eval_metric='logloss')
    clf3 = LogisticRegression(random_state=42)

    ensemble_model = VotingClassifier(estimators=[('rf', clf1), ('xgb', clf2), ('lr', clf3)], voting='soft')
    ensemble_model.fit(X_train, y_train_encoded)

    return ensemble_model

def evaluate_model(model, X_test, y_test):
    """
    evaluate_model: Evaluate the trained model on the test data.
    """
    y_pred_encoded = model.predict(X_test)

    label_encoder = LabelEncoder()
    label_encoder.fit(['Red', 'Blue']) 
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    y_test_decoded = label_encoder.inverse_transform(y_test)

    print(f"Accuracy: {accuracy_score(y_test_decoded, y_pred)}")
    print("Classification Report:")
    print(classification_report(y_test_decoded, y_pred))

def get_fighter_averages(df, fighter_name: str, color_prefix: str):
    red_cols = [col for col in df.columns if col.startswith("Red")]
    blue_cols = [col for col in df.columns if col.startswith("Blue")]

    red_df = df[df["RedFighter"] == fighter_name][red_cols].copy()
    blue_df = df[df["BlueFighter"] == fighter_name][blue_cols].copy()

    # Normalize column names
    blue_df.columns = [col.replace("Blue", "Red", 1) for col in blue_df.columns]

    all_fights = pd.concat([red_df, blue_df], ignore_index=True)

    return all_fights.mean(numeric_only=True)

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)

def main():
    os.makedirs("C:/Code/ddi_course/capstone_project/model", exist_ok=True)

    numerical_columns = [
        'RedOdds', 'BlueOdds', 'RedExpectedValue', 'BlueExpectedValue', 
        'BlueAvgSigStrLanded', 'BlueAvgSigStrPct', 'BlueAvgSubAtt', 'BlueAvgTDLanded', 
        'BlueAvgTDPct', 'BlueLongestWinStreak', 'BlueLosses', 'BlueTotalRoundsFought', 
        'BlueTotalTitleBouts', 'BlueWinsByDecisionMajority', 'BlueWinsByDecisionSplit', 
        'BlueWinsByDecisionUnanimous', 'BlueWinsByKO', 'BlueWinsBySubmission', 
        'BlueWinsByTKODoctorStoppage', 'BlueWins', 'BlueHeightCms', 'BlueReachCms', 
        'BlueWeightLbs', 'RedCurrentLoseStreak', 'RedCurrentWinStreak', 'RedDraws', 
        'RedAvgSigStrLanded', 'RedAvgSigStrPct', 'RedAvgSubAtt', 'RedAvgTDLanded', 
        'RedAvgTDPct', 'RedLongestWinStreak', 'RedLosses', 'RedTotalRoundsFought', 
        'RedTotalTitleBouts', 'RedWinsByDecisionMajority', 'RedWinsByDecisionSplit', 
        'RedWinsByDecisionUnanimous', 'RedWinsByKO', 'RedWinsBySubmission', 
        'RedWinsByTKODoctorStoppage', 'RedWins', 'RedHeightCms', 'RedReachCms', 
        'RedWeightLbs', 'RedAge', 'BlueAge', 'LoseStreakDif', 'WinStreakDif', 
        'LongestWinStreakDif', 'WinDif', 'LossDif', 'TotalRoundDif', 'TotalTitleBoutDif', 
        'KODif', 'SubDif', 'HeightDif', 'ReachDif', 'AgeDif', 'SigStrDif', 'AvgSubAttDif', 
        'AvgTDDif', 'BMatchWCRank', 'RMatchWCRank', 'RWFlyweightRank', 'RWFeatherweightRank', 
        'RWStrawweightRank', 'RWBantamweightRank', 'RHeavyweightRank', 'RLightHeavyweightRank', 
        'RMiddleweightRank', 'RWelterweightRank', 'RLightweightRank', 'RFeatherweightRank', 
        'RBantamweightRank', 'RFlyweightRank', 'RPFPRank', 'BWFlyweightRank', 'BWFeatherweightRank', 
        'BWStrawweightRank', 'BWBantamweightRank', 'BHeavyweightRank', 'BLightHeavyweightRank', 
        'BMiddleweightRank', 'BWelterweightRank', 'BLightweightRank', 'BFeatherweightRank', 
        'BBantamweightRank', 'BFlyweightRank', 'BPFPRank', 'FinishRound',
        'RedDecOdds', 'BlueDecOdds', 'RSubOdds', 'BSubOdds', 'RKOOdds', 'BKOOdds'
    ]

    categorical_columns = [
        'WeightClass', 'Gender', 'BlueStance', 'RedStance', 'BetterRank'
    ]
    data = pd.read_csv("C:/Code/ddi_course/capstone_project/data/ufc-master.csv")
    MODEL_COLUMNS = numerical_columns + categorical_columns
    
    X_raw = data[MODEL_COLUMNS]
    y_raw = data["Winner"]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    X_train, scaler, label_encoders = preprocess_input_train(X_train_raw, numerical_columns, categorical_columns)
    X_test = preprocess_input_test(X_test_raw, scaler, label_encoders, numerical_columns, categorical_columns)

    model = tune_models(X_train, y_train_encoded)

    evaluate_model(model, X_test, y_test_encoded)

    joblib.dump(model, "C:/Code/ddi_course/capstone_project/models/ufc_model_v5.pkl")
    joblib.dump(scaler, "C:/Code/ddi_course/capstone_project/model/scaler_v1.pkl")
    joblib.dump(label_encoder, "C:/Code/ddi_course/capstone_project/model/label_encoder_v1.pkl")
    joblib.dump(label_encoders, "C:/Code/ddi_course/capstone_project/model/label_encoders_dict_v1.pkl")
    joblib.dump(X_train.columns.tolist(), "C:/Code/ddi_course/capstone_project/model/feature_columns_v1.pkl")

if __name__ == "__main__":
     main()

