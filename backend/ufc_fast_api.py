""" 
This will act as the back end API for the UFC prdiction model application run throught the FastAPI framework.
The front end will be run through the streamlit framework.
"""
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import src.function as f
from sklearn.preprocessing import LabelEncoder
import traceback
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from sklearn.exceptions import NotFittedError
from typing import Optional

app = FastAPI()

model = joblib.load("C:/Code/capstone_project/backend/models/ufc_model_v5.pkl")
scaler = joblib.load("C:/Code/capstone_project/backend/model/scaler_v1.pkl")
label_encoder = joblib.load("C:/Code/capstone_project/backend/model/label_encoder_v1.pkl")
label_encoders = joblib.load("C:/Code/capstone_project/backend/model/label_encoders_dict_v1.pkl")
feature_columns = joblib.load("C:/Code/capstone_project/backend/model/feature_columns_v1.pkl")
df = pd.read_csv("C:/Code/capstone_project/backend/data/ufc-master.csv")

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


class FighterStats(BaseModel):
    features: dict
    red_fighter_name: Optional[str] = None
    blue_fighter_name: Optional[str] = None

@app.get("/")
def read_root():
    return {"message": "Welcome to the UFC Prediction API!"}

@app.get("/data")
def return_data():
    file_path = "C:/Code/capstone_project/backend/data/ufc-master.csv"
    return FileResponse(file_path, media_type='text/csv', filename="ufc-master.csv")

@app.get("/test_predictions")
def get_test_predictions():
    test_set = pd.read_csv("data/upcoming_predictions.csv")
    return test_set.to_dict(orient="records")

@app.get("/model_predictions")
def get_model_predictions():
    try:
        df = pd.read_csv("C:/Code/capstone_project/backend/data/ufc-master.csv")

        output_columns = ['RedFighter', 'BlueFighter', 'Winner']

        required_features = [
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
            'RedDecOdds', 'BlueDecOdds', 'RSubOdds', 'BSubOdds', 'RKOOdds', 'BKOOdds', 
            'WeightClass', 'Gender', 'BlueStance', 'RedStance', 'BetterRank',
            'RedFighter', 'BlueFighter', 'Winner'  # ✅ Include for output
        ]

        new_df = df[[col for col in required_features if col in df.columns]]

        for col in new_df.columns:
            if new_df[col].dtype in ['int64', 'float64']:
                new_df[col] = new_df[col].fillna(0)
            elif new_df[col].dtype == 'object':
                new_df[col] = new_df[col].fillna('Unknown')

        y = new_df['Winner']
        X_raw = new_df.drop(columns=['Winner', 'RedFighter', 'BlueFighter'])

        numerical_columns = X_raw.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_columns = X_raw.select_dtypes(include=['object']).columns.tolist()

        X, _, _ = f.preprocess_input_train(X_raw, numerical_columns, categorical_columns)
        y_pred = model.predict(X)

        label_encoder = LabelEncoder()
        label_encoder.fit(['Red', 'Blue'])
        decoded_preds = label_encoder.inverse_transform(y_pred)
        decoded_actuals = label_encoder.transform(y)  # Optional: make sure labels are consistent

        new_df['PredictedWinner'] = decoded_preds
        new_df['Winner'] = label_encoder.inverse_transform(decoded_actuals)

        result_df = new_df[['RedFighter', 'BlueFighter', 'Winner', 'PredictedWinner']].astype(str)
        return result_df.to_dict(orient="records")
    
    except Exception as e:
        return {"error": str(e)}
    
# @app.get("/model_predictions")
# def get_model_predictions():
#     try:
#         df = pd.read_csv("C:/Code/ddi_course/capstone_project/backend/data/ufc-master.csv")
#         new_df = df.drop(columns=['BlueCurrentLosesStreak', 'BlueCurrentWinStreak','BlueDraws', 'EmptyArena', 'NumberOfRounds'], errors='ignore')
#         for col in new_df.columns:
#             if new_df[col].dtype in ['int64', 'float64']:
#                 new_df[col] = new_df[col].fillna(0)
#             elif new_df[col].dtype == 'object':
#                 new_df[col] = new_df[col].fillna('Unknown')
    
#         #new_df = new_df.drop(columns=['PredictedWinner'])
#         X, y = f.preprocess_input_train(new_df)
#         y_pred = model.predict(X)
        
#         label_encoder = LabelEncoder()
#         label_encoder.fit(['Red', 'Blue'])

#         decoded_preds = label_encoder.inverse_transform(y_pred)
#         decoded_actuals = y.values  

#         new_df['PredictedWinner'] = decoded_preds
#         new_df['Winner'] = decoded_actuals
        
#         model_prdic = new_df[['RedFighter', 'BlueFighter', 'Winner', 'PredictedWinner']].copy()
#         model_prdic = model_prdic.astype(str)
#         return model_prdic.to_dict(orient="records")
#         #model_prdic = new_df[['RedFighter', 'BlueFighter', 'Winner', 'PredictedWinner']].to_dict(orient="records")
#         #return model_prdic.to_dict(orient="records")
#         #return list(X.columns)
#     except Exception as e:
#         return {"error model pred endpoint": str(e), "x columns": X.columns}

@app.post("/predict")
def predict(stats: FighterStats):
    input_data = stats.features
    model_input_data = {}

    red_fighter_row = df[df["RedFighter"] == stats.red_fighter_name].mean(numeric_only=True) if stats.red_fighter_name else {}
    blue_fighter_row = df[df["BlueFighter"] == stats.blue_fighter_name].mean(numeric_only=True) if stats.blue_fighter_name else {}

    for col in feature_columns:
        if col in input_data:
            model_input_data[col] = input_data[col]
        elif col.startswith("Red") and col in red_fighter_row:
            model_input_data[col] = red_fighter_row[col]
        elif col.startswith("Blue") and col in blue_fighter_row:
            model_input_data[col] = blue_fighter_row[col]
        elif col in df.columns:
            model_input_data[col] = df[col].mean() if df[col].dtype != "O" else df[col].mode()[0]
        else:
            model_input_data[col] = 0  # fallback

    # Preprocessing
    model_input_df = pd.DataFrame([model_input_data])
    model_input_df[numerical_columns] = scaler.transform(model_input_df[numerical_columns])

    for col in categorical_columns:
        if col in model_input_df.columns:
            le = label_encoders[col]
            val = model_input_df[col].values[0]
            if val not in le.classes_:
                val = 'Unknown'
                if 'Unknown' not in le.classes_:
                    le.classes_ = np.append(le.classes_, 'Unknown')
            model_input_df[col] = le.transform([val])

    model_input_df = model_input_df.reindex(columns=feature_columns)

    prediction_encoded = model.predict(model_input_df)[0]
    prediction = label_encoder.inverse_transform([prediction_encoded])[0]

    return {"prediction": prediction}

# @app.post("/predict")
# def predict(stats: FighterStats):
#     input_data = stats.features
#     model_input_data = {}

#     for col in feature_columns:
#         if col in input_data:
#             model_input_data[col] = input_data[col]
#         elif col in df.columns:
#             model_input_data[col] = df[col].mean() if df[col].dtype != "O" else df[col].mode()[0]
#         else:
#             model_input_data[col] = 0

#     model_input_df = pd.DataFrame([model_input_data])

#     model_input_df[numerical_columns] = scaler.transform(model_input_df[numerical_columns])

#     for col in categorical_columns:
#         if col in model_input_df.columns:
#             le = label_encoders[col]
#             val = model_input_df[col].values[0]
#             if val not in le.classes_:
#                 val = 'Unknown'
#                 if 'Unknown' not in le.classes_:
#                     le.classes_ = np.append(le.classes_, 'Unknown')
#             model_input_df[col] = le.transform([val])

#     model_input_df = model_input_df.reindex(columns=feature_columns)

#     prediction_encoded = model.predict(model_input_df)[0]
#     prediction = label_encoder.inverse_transform([prediction_encoded])[0]

#     return {"prediction": prediction}