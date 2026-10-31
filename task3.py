
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

sns.set_theme(style="whitegrid")
os.makedirs("outputs", exist_ok=True)

# 1. Load Dataset
print("Loading Bank Marketing Dataset...")

df = pd.read_csv("bank-full.csv", sep=";")

print("\nDataset Loaded Successfully!")
print("\nFirst Five Rows:")
print(df.head())

# 2. Explore Dataset
print("\nDataset Shape:", df.shape)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

# 3. Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# 4. Data Cleaning
# Convert unknown values represented by '?' into missing values
df.replace("unknown", np.nan, inplace=True)

# Remove duplicate records
df.drop_duplicates(inplace=True)

print("\nDataset Shape After Cleaning:", df.shape)

print("\nTarget Variable Distribution:")
print(df["y"].value_counts())

# 5. Prepare Features and Target
# Exclude duration to avoid data leakage
X = df.drop(columns=["y", "duration"])
y = df["y"].map({"no": 0, "yes": 1})

# Identify numerical and categorical columns
numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical Features:", numerical_features)
print("\nCategorical Features:", categorical_features)

# 6. Preprocessing
numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

# 7. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# 8. Build Decision Tree Classifier
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        criterion="gini",
        max_depth=6,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42
    ))
])

# 9. Train Model
print("\nTraining Decision Tree Classifier...")
model.fit(X_train, y_train)

print("Model Training Completed Successfully!")

# 10. Make Predictions
y_pred = model.predict(X_test)

# 11. Model Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")
print("Model Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
report = classification_report(
    y_test,
    y_pred,
    target_names=["No Subscription", "Yes Subscription"]
)
print(report)

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Save evaluation report
with open("outputs/model_evaluation.txt", "w") as file:
    file.write("DECISION TREE MODEL EVALUATION\n\n")
    file.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
    file.write(report)
    file.write("\nConfusion Matrix:\n")
    file.write(str(cm))

# 12. Confusion Matrix Visualization
plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No", "Yes"],
    yticklabels=["No", "Yes"]
)
plt.title("Decision Tree - Confusion Matrix")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=300)
plt.show()

# 13. Decision Tree Visualization
tree_model = model.named_steps["classifier"]
feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()

plt.figure(figsize=(24, 12))
plot_tree(
    tree_model,
    feature_names=feature_names,
    class_names=["No", "Yes"],
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=8
)
plt.title("Decision Tree Classifier")
plt.savefig("outputs/decision_tree.png", dpi=300)
plt.show()

# 14. Feature Importance
importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": tree_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
).head(10)

print("\nTop 10 Important Features:")
print(importance)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=importance,
    x="Importance",
    y="Feature"
)
plt.title("Top 10 Important Features")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=300)
plt.show()

# 15. Final Result
print("\n====================================")
print("PRODIGY TASK 03 COMPLETED!")
print("Decision Tree Model Trained Successfully!")
print("Predictions and Visualizations Saved!")
print("====================================")
