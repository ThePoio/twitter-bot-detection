import pandas as pd
from pipeline_data import process_df

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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
    r".data/bot_detection_data.csv"
)

df = process_df(df)

y = LabelEncoder().fit_transform(df["Bot Label"])

X = df[[
    "Tweet",
    "Retweet Count",
    "Mention Count",
    "Follower Count",
    "account_age_days",
    "has_hashtags",
    "tweet_length",
    "Verified",
    "key_words"
]]

X = X.copy()
X["Tweet"] = X["Tweet"].fillna("")
numeric_columns = [
    "Retweet Count",
    "Mention Count",
    "Follower Count",
    "account_age_days",
    "has_hashtags",
    "tweet_length",
    "Verified",
    "key_words"
]
X[numeric_columns] = X[numeric_columns].fillna(0)

pre = ColumnTransformer(
    transformers=[
        ("tweet", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        ), "Tweet")
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
    ("clf", SVC(
        kernel="linear",
        C=2,
        class_weight="balanced"
    ))
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
        n_estimators=500,
        random_state=42,
        max_depth=24,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1
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