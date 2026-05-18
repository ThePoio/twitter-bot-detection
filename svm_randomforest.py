import pandas as pd
from pipeline_data import process_df

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

df = pd.read_csv(
    r"C:\Users\yaelz\proyectom\twitter-bot-detection\bot_detection_data.csv"
)

df = process_df(df)

y = LabelEncoder().fit_transform(df["Bot Label"])

X = df[[
    "Tweet",
    "Retweet Count",
    "Mention Count",
    "Follower Count",
    "account_age_days",
    "username_digits",
    "has_hashtags",
    "tweet_length",
    "url_count"
]]

X = X.fillna("")

pre = ColumnTransformer(
    transformers=[
        ("tweet", TfidfVectorizer(max_features=3000), "Tweet")
    ],
    remainder="passthrough"
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===== SVM =====
svm = Pipeline([
    ("pre", pre),
    ("clf", SVC(kernel="rbf", C=10))
])

svm.fit(X_train,y_train)

pred_svm = svm.predict(X_test)

print("\n===== SVM =====")
print("Accuracy:",accuracy_score(y_test,pred_svm))
print("Precision:",precision_score(y_test,pred_svm))
print("Recall:",recall_score(y_test,pred_svm))
print("F1:",f1_score(y_test,pred_svm))
print(confusion_matrix(y_test,pred_svm))
print(classification_report(y_test,pred_svm))


# ===== RF =====
rf = Pipeline([
    ("pre", pre),
    ("clf", RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ))
])

rf.fit(X_train,y_train)

pred_rf = rf.predict(X_test)

print("\n===== RANDOM FOREST =====")
print("Accuracy:",accuracy_score(y_test,pred_rf))
print("Precision:",precision_score(y_test,pred_rf))
print("Recall:",recall_score(y_test,pred_rf))
print("F1:",f1_score(y_test,pred_rf))
print(confusion_matrix(y_test,pred_rf))
print(classification_report(y_test,pred_rf))