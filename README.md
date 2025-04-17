# Galvanize DDI Cohort 11 Capstone Project: UFC Prediction

## Background & Purpose

This project was created as my final capstone for the **Galvanize DDI Cohort 11** course. I chose this dataset because I believed it would be a great evaluation of the skills I’ve learned. As an MMA fan, it was exciting to work with real UFC data and explore ways to predict fight outcomes.

---

## Dataset Info

**Source:**  
[Ultimate UFC Dataset on Kaggle](https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset/data)  
This dataset merges all public UFC datasets on Kaggle into a comprehensive dataset ideal for machine learning and fight analytics.

---

## Data Exploration

While EDA wasn't my primary focus (since I already had a prediction goal in mind), I still explored a few key areas:

### 🟥🟦 Red vs. Blue Wins
![Red vs Blue Wins](/images/red_blue_win.png)

### 💥 Fight Finish Types
![Finish Types](//images/Finish_type.png)

### 📏 Winner Statistic Differences (e.g., Height)
![Winner Height Difference](/images/winner_height_dif.png)

---

## Data Processing

To prepare the data for modeling, I cleaned and scaled numerical features and encoded categorical ones using `LabelEncoder`.

```python
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
```
## Hypothesis Testing

**Question:**  
Do red fighters win more often than blue fighters?

### Hypotheses

- **H₀ (Null):** Win rates are equal for red and blue corners.  
- **H₁ (Alternative):** Win rates are different.

I performed a **proportional Z-test** on the dataset to evaluate this hypothesis.

- **Z-statistic:** 18.309  
- **P-value:** 0.000  

✅ **Conclusion:** I reject the null hypothesis. There is a statistically significant difference in win rates — red fighters win more often.

---

## Model Building

I went through five iterations of models and landed on a **VotingClassifier** which incorporates:

- `RandomForestClassifier`  
- `XGBClassifier`  
- `LogisticRegression`

---

### Final Model Scores

![Model V5](/images/model_v5.png)

---

### Test Set Performance

![Confusion Matrix - Test Set](/images/Confusion_matrix_test_set.png)

- ✅ Accuracy: **87%**  
- 🎯 Precision: **90%**  
- 🔁 Recall: **87%**  
- 🏅 F1 Score: **89%**

---

### Independent Standalone Dataset

![Confusion Matrix - Standalone Test](/images/Confusion_matrix_stand_alone_test.png)

- ✅ Accuracy: **62%**  
- 🎯 Precision: **71%**  
- 🔁 Recall: **62%**  
- 🏅 F1 Score: **67%**

---

## Further Development

I dropped some columns due to modeling constraints, but given more time, I would like to include the following features into the model:

- `Location`  
- `Country`  
- `TitleBout`  
- `FinishDetails`  
- `Finish`  
- `TotalFightTimeSecs`  
- `FinishRoundTime`

---

## Conclusion

The model is more effective when predicting **Red Fighter** outcomes but still performs reasonably well for **Blue Fighter** predictions. With more time and research, there's plenty of room for improvement.

Adding more features helped improve performance, and I plan to keep refining the model. I’m also aiming to use it to predict real upcoming UFC fights — and maybe even enter it into some coding or data science competitions!



