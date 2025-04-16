import streamlit as st
import requests
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
import io
import plotly.express as px

st.title("🥊 UFC Fight Outcome Prediction Tool")

st.sidebar.header("Menu")
page = st.sidebar.selectbox("Choose a Page", ["Starting Data", "Test Set Predictions", "Predict an Upcoming Fight"])

API_URL = "http://localhost:8000"  

if page == "Starting Data":
    
    data_response = requests.get(f"{API_URL}/data")
    if data_response.status_code == 200:
        df = pd.read_csv(io.StringIO(data_response.text)) 

        st.header("📊 Explore Your Data")

        winner_counts = df['Winner'].value_counts().reset_index()
        winner_counts.columns = ['Winner', 'Count']

        red_wins = df['Winner'].value_counts().get('Red', 0)
        blue_wins = df['Winner'].value_counts().get('Blue', 0)

        st.write(f"Red Wins: {red_wins}")
        st.write(f"Blue Wins: {blue_wins}")
        chart = alt.Chart(winner_counts).mark_bar().encode(
            x=alt.X('Winner:N', title='Fight Winner'),
            y=alt.Y('Count:Q', title='Number of Wins'),
            color=alt.Color('Winner:N', scale=alt.Scale(scheme='tableau10')),
            tooltip=['Winner', 'Count']
        ).properties(
            title='Distribution of Fight Winners',
            width=600,
            height=400
        )

        st.altair_chart(chart, use_container_width=True)

        if st.checkbox("Show DataFrame"):
            st.write("Here’s a preview of your data:")
            st.dataframe(df.head(10))

        col1, col2 = st.columns(2)
        with col1:
            if st.checkbox("Show column types"):
                st.write(df.dtypes)
        with col2:
            if st.checkbox("Show summary statistics"):
                st.write(df.describe())

    else:
        st.error("Failed to fetch data from the backend API. Please check the /data endpoint or your API server.")


elif page == "Test Set Predictions":
    st.header("📈 Test Set Results and Verification")

    tab1, tab2 = st.tabs(["Test Set Predictions", "Model Predictions"])

    with tab1:
        response = requests.get(f"{API_URL}/test_predictions")
        if response.status_code == 200:
            test_set = pd.DataFrame(response.json())
            test_set["Correct"] = test_set["PredictedWinner"] == test_set["Winner"]
            if st.checkbox("Show Test Set DataFrame"):
                st.dataframe(test_set)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("✅ Prediction Accuracy")
                counts = test_set["Correct"].value_counts()
                counts.index = ["Correct", "Incorrect"]
                st.write("Correct predictions: ", counts["Correct"])
                st.write("Incorrect predictions: ", counts["Incorrect"])
                percent_correct = counts["Correct"] / (counts["Correct"] + counts["Incorrect"]) * 100
                st.write(f"Percent: {percent_correct:.2f}%")
                st.bar_chart(counts)
            with col2:
                st.subheader("📊 Model Performance")
                red_correct = len(test_set[(test_set["PredictedWinner"] == "Red") & (test_set["Correct"] == True)])
                red_incorrect = len(test_set[(test_set["PredictedWinner"] == "Red") & (test_set["Correct"] == False)])
                blue_correct = len(test_set[(test_set["PredictedWinner"] == "Blue") & (test_set["Correct"] == True)])
                blue_incorrect = len(test_set[(test_set["PredictedWinner"] == "Blue") & (test_set["Correct"] == False)])

                st.write(f"Red - Correct: {red_correct}, Red - Incorrect: {red_incorrect}, Prercent: {red_correct/(red_correct+red_incorrect)*100:.2f}%")
                st.write(f"Blue - Correct: {blue_correct}, Blue - Incorrect: {blue_incorrect}, Prercent: {blue_correct/(blue_correct+blue_incorrect)*100:.2f}%")
                fig = px.histogram(test_set, x="PredictedWinner", color="Correct", barmode="group", title="Model Predictions")
                fig.update_layout(xaxis_title="Predicted Winner", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
                
            st.subheader("📊 Model Performance by Fighter")
            if st.checkbox("Show Confusion Matrix"):
                fig, ax = plt.subplots()
                cm = confusion_matrix(test_set["Winner"], test_set["PredictedWinner"], labels=["Red", "Blue"])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Red", "Blue"], yticklabels=["Red", "Blue"], ax=ax)
                ax.set_xlabel("Predicted Winner")
                ax.set_ylabel("Actual Winner")
                st.pyplot(fig)
        else:
            st.error("Failed to fetch predictions.")
    
    with tab2:

        other_response = requests.get(f"{API_URL}/model_predictions")
        if other_response.status_code == 200:
            if isinstance(other_response, dict):
                other_response = [{"key": key, "value": value} for key, value in other_response.items()]
            model_set = pd.DataFrame(other_response.json())
            #st.write(model_set)
            model_set["Correct"] = model_set["PredictedWinner"] == model_set["Winner"]
            #st.dataframe(model_set)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("✅ Prediction Accuracy")
                counts = model_set["Correct"].value_counts()
                counts.index = ["Correct", "Incorrect"]
                st.write("Correct predictions: ", counts["Correct"])
                st.write("Incorrect predictions: ", counts["Incorrect"])
                percent_correct = counts["Correct"] / (counts["Correct"] + counts["Incorrect"]) * 100
                st.write(f"Percent: {percent_correct:.2f}%")
                st.bar_chart(counts)
            with col2:
                st.subheader("📊 Model Performance")
                red_correct = len(model_set[(model_set["PredictedWinner"] == "Red") & (model_set["Correct"] == True)])
                red_incorrect = len(model_set[(model_set["PredictedWinner"] == "Red") & (model_set["Correct"] == False)])
                blue_correct = len(model_set[(model_set["PredictedWinner"] == "Blue") & (model_set["Correct"] == True)])
                blue_incorrect = len(model_set[(model_set["PredictedWinner"] == "Blue") & (model_set["Correct"] == False)])

                st.write(f"Red - Correct: {red_correct}, Red - Incorrect: {red_incorrect}, Prercent: {red_correct/(red_correct+red_incorrect)*100:.2f}%")
                st.write(f"Blue - Correct: {blue_correct}, Blue - Incorrect: {blue_incorrect}, Prercent: {blue_correct/(blue_correct+blue_incorrect)*100:.2f}%")
                fig = px.histogram(model_set, x="PredictedWinner", color="Correct", barmode="group", title="Model Predictions")
                fig.update_layout(xaxis_title="Predicted Winner", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("📊 Model Performance by Fighter")
            if st.checkbox("Trained Confusion Matrix"):
                fig, ax = plt.subplots()
                cm = confusion_matrix(model_set["Winner"], model_set["PredictedWinner"], labels=["Red", "Blue"])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Red", "Blue"], yticklabels=["Red", "Blue"], ax=ax)
                ax.set_xlabel("Predicted Winner")
                ax.set_ylabel("Actual Winner")
                st.pyplot(fig)
        else:
            print(f"Failed to fetch predictions. Status code: {other_response.status_code}")
            print(other_response.text)

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

    col1, col2 = st.columns(2)

    red_input = {}
    blue_input = {}

    data_response = requests.get(f"{API_URL}/data")
    if data_response.status_code == 200:
        df = pd.read_csv(io.StringIO(data_response.text)) 

    with col1:
        st.markdown("### 🔴 Red Fighter")
        for feature in [f for f in important_features if f.startswith("Red")]:
            red_input[feature] = st.number_input(
                important_features[feature],
                min_value=float(df[feature].min()),
                max_value=float(df[feature].max()),
                value=float(df[feature].mean()),
                step=1.0
            )

    with col2:
        st.markdown("### 🔵 Blue Fighter")
        for feature in [f for f in important_features if f.startswith("Blue")]:
            blue_input[feature] = st.number_input(
                important_features[feature],
                min_value=float(df[feature].min()),
                max_value=float(df[feature].max()),
                value=float(df[feature].mean()),
                step=1.0,
                key=f"blue_{feature}"
            )

    if st.button("Predict Winner"):
        all_inputs = {**red_input, **blue_input}
        response = requests.post(f"{API_URL}/predict", json={"features": all_inputs})

        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.success(f"🏆 Predicted Winner: **{prediction}** Fighter")
        else:
            st.error("Prediction failed. Please try again.")