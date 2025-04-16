import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    df = pd.read_csv("C:/Code/ddi_course/capstone_project/data/ufc-master.csv")
    return df


model = joblib.load("C:/Code/ddi_course/capstone_project/models/ufc_model_v3.pkl")
scaler = joblib.load("C:/Code/ddi_course/capstone_project/model/scaler.pkl")
label_encoder = joblib.load("C:/Code/ddi_course/capstone_project/model/label_encoder.pkl")
label_encoders = joblib.load("C:/Code/ddi_course/capstone_project/model/label_encoders_dict.pkl")
feature_columns = joblib.load("C:/Code/ddi_course/capstone_project/model/feature_columns.pkl")

st.title("🥊 UFC Fight Outcome Prediction Tool")

st.sidebar.header("Menu")
page = st.sidebar.selectbox("Choose a Page", ["Starting Data", "Test Set Predictions", "Predict an Upcoming Fight"])

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

df = load_data()

if page == "Starting Data":
    st.header("📊 Explore Your Data")
    if df is not None:
        st.write("Here’s a preview of your data:")
        st.dataframe(df.head(10))
    col1, col2 = st.columns(2)
    with col1:
        if st.checkbox("Show column types"):
            st.write(df.dtypes)
    with col2:
        if st.checkbox("Show summary statistics"):
            st.write(df.describe())


elif page == "Test Set Predictions":
    st.header("📈 Test Set Results and Verification")

    # Load your data
    test_set = pd.read_csv("C:/Code/ddi_course/capstone_project/data/upcoming_predictions.csv")

    # Make sure column names are stripped of whitespace (optional cleanup)
    test_set.columns = test_set.columns.str.strip()

    # Create a column for whether prediction was correct
    test_set["Correct"] = test_set["PredictedWinner"] == test_set["Winner"]

    # Show the dataframe (optional)
    st.subheader("📋 Fight Predictions")
    st.dataframe(test_set)

    # Plot 1: Accuracy Bar Chart
    st.subheader("✅ Prediction Accuracy: 5 Red / 3 Blue")
    import matplotlib.pyplot as plt

    fig1, ax1 = plt.subplots()
    accuracy_counts = test_set["Correct"].value_counts()
    accuracy_counts.index = ["Correct", "Incorrect"]
    accuracy_counts.plot(kind="bar", color=["green", "red"], ax=ax1)
    ax1.set_ylabel("Number of Fights")
    ax1.set_title("Prediction Accuracy")
    st.pyplot(fig1)

    # Plot 2: Confusion Matrix (optional)
    if st.checkbox("Show Confusion Matrix"):
        st.subheader("🔀 Confusion Matrix")
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        fig2, ax2 = plt.subplots()
        labels = ["Red", "Blue"]
        cm = confusion_matrix(test_set["Winner"], test_set["PredictedWinner"], labels=labels)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax2)
        ax2.set_xlabel("Predicted Winner")
        ax2.set_ylabel("Actual Winner")
        ax2.set_title("Predicted vs Actual")
        st.pyplot(fig2)

    


elif page == "Predict an Upcoming Fight":
   
    st.header("🔮 Predict an Upcoming Fight")
    st.write("Enter fighter stats to make a prediction:")

    important_features = {
    "RedWinsByKO": "Red - Wins by Knockout",
    "RedWinsBySubmission": "Red - Wins by Submission",
    "RedAvgSigStrPct": "Red - Striking Accuracy (%)",
    "RedAvgSigStrLanded": "Red - Sig. Strikes Landed",
    "RedAvgTDLanded": "Red - Takedowns Landed",
    "RedAvgTDPct": "Red - Takedown Accuracy (%)",
    "RedAvgSubAtt": "Red - Submission Attempts",


    "BlueWinsByKO": "Blue - Wins by Knockout",
    "BlueWinsBySubmission": "Blue - Wins by Submission",
    "BlueAvgSigStrPct": "Blue - Striking Accuracy (%)",
    "BlueAvgSigStrLanded": "Blue - Sig. Strikes Landed",
    "BlueAvgTDLanded": "Blue - Takedowns Landed",
    "BlueAvgTDPct": "Blue - Takedown Accuracy (%)",
    "BlueAvgSubAtt": "Blue - Submission Attempts"
    }

    st.subheader("👊 Enter Fighter Stats")

    col1, col2 = st.columns(2)

    
    with col1:
        st.markdown("### 🔴 Red Fighter")
        red_input = {}
        for feature in [f for f in important_features if f.startswith("Red")]:
        
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            avg_val = float(df[feature].mean())

            red_input[feature] = st.number_input(
                important_features[feature],
                min_value=min_val,
                max_value=max_val,
                value=avg_val,
                step=(max_val - min_val) / 100
            )

    with col2:
        st.markdown("### 🔵 Blue Fighter")
        blue_input = {}
        for feature in [f for f in important_features if f.startswith("Blue")]:
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            avg_val = float(df[feature].mean())

            blue_input[feature] = st.number_input(
                important_features[feature],
                min_value=min_val,
                max_value=max_val,
                value=avg_val,
                step=(max_val - min_val) / 100,
                key=f"blue_{feature}"  
            )

    user_input = {**red_input, **blue_input}


    all_required_columns = model.feature_names_in_  

    model_input_data = {}
    for col in all_required_columns:
        if col in df.columns:
            if df[col].dtype == "O":
                model_input_data[col] = df[col].mode()[0]
            else:
                model_input_data[col] = df[col].mean()
        else:
            model_input_data[col] = 0

    model_input_data.update(user_input)

    model_input_df = pd.DataFrame([model_input_data])


    model_input_df[numerical_columns] = scaler.transform(model_input_df[numerical_columns])

    for col in categorical_columns:
        if col in model_input_df.columns:
            le = label_encoders[col]
            val = model_input_df[col].values[0]

            if val not in le.classes_:
                st.warning(f"Value '{val}' for '{col}' not seen during training. Setting to 'Unknown'.")
                val = 'Unknown'
                if 'Unknown' not in le.classes_:
                    le.classes_ = np.append(le.classes_, 'Unknown')

            model_input_df[col] = le.transform([val])

    model_input_df = model_input_df.reindex(columns=feature_columns)

    prediction_encoded = model.predict(model_input_df)[0]
    prediction = label_encoder.inverse_transform([prediction_encoded])[0]

    st.success(f"Prediction: 🥊 {prediction} Fighter Wins!")


