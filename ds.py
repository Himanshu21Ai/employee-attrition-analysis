import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import seaborn as sns;
from sklearn.model_selection import train_test_split;
from sklearn.preprocessing import StandardScaler;
from sklearn.linear_model import LogisticRegression;
from sklearn.ensemble import RandomForestClassifier;
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
);

df = pd.read_csv("dataset/IBM.csv")

# datascience
# checck which column have what type of catgory and how much category
cat_cols = df.select_dtypes(include="object").columns
print("Categorical Columns:")
print(df.select_dtypes(include="object").columns)
cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    print(f"{col} --> {df[col].nunique()} categories")
for col in cat_cols:
    print(f"\n{col}")
    print(df[col].unique())

for col in df.select_dtypes(include="int64").columns:
    print(f"\n{col}")
    print("Unique values:", df[col].nunique())
    print(df[col].unique())

# encoding the features means labeling and onehot lableing
# as attirtion have only two output so label encoding
df["Attrition"] = df["Attrition"].map({"Yes":1,"No":0})

#as this three column have multiple category so on hot encoding
df = pd.get_dummies(
    df,
    columns=[
       
        "Department",
        "EducationField",
        "MaritalStatus"
    ],
    drop_first=True
)


# #feature selection tageting = select target (output ) and featuring(input)
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

#ml
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


#scaling the feature so that it cant pamper the model result
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
print("Scaling completed successfully")
print(X_train.shape)
print(X_test.shape)
print(X_train[:2])

# Create Random Forest model
model_RF = RandomForestClassifier(
    random_state=42,
    class_weight="balanced"
)

# Train Random Forest
model_RF.fit(X_train, y_train)

# Prediction
y_pred_rf = model_RF.predict(X_test)


# ================= BALANCED LOGISTIC REGRESSION =================

# Create Logistic Regression model
logistic_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

# Train Logistic Regression
logistic_model.fit(X_train, y_train)

# Prediction
y_pred_lr = logistic_model.predict(X_test)


# ================= MODEL COMPARISON =================

# Random Forest metrics
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)


# Logistic Regression metrics
lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_precision = precision_score(y_test, y_pred_lr)
lr_recall = recall_score(y_test, y_pred_lr)
lr_f1 = f1_score(y_test, y_pred_lr)


# Display comparison
print("\n========== MODEL COMPARISON ==========")

print("\nRandom Forest")
print("Accuracy :", rf_accuracy)
print("Precision:", rf_precision)
print("Recall   :", rf_recall)
print("F1 Score :", rf_f1)

print("\nBalanced Logistic Regression")
print("Accuracy :", lr_accuracy)
print("Precision:", lr_precision)
print("Recall   :", lr_recall)
print("F1 Score :", lr_f1)