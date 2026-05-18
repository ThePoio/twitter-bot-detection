import pandas as pd
from pipeline_data import process_df

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
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
    remainder="passthrough",
    sparse_threshold=0.0
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===== DECISION TREE =====
tree = Pipeline([
    ("pre", pre),
    ("clf", DecisionTreeClassifier(
        random_state=42,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight="balanced"
    ))
])

tree.fit(X_train, y_train)

pred_tree = tree.predict(X_test)

print("\n===== DECISION TREE =====")
print("Accuracy:", accuracy_score(y_test, pred_tree))
print("Precision:", precision_score(y_test, pred_tree))
print("Recall:", recall_score(y_test, pred_tree))
print("F1:", f1_score(y_test, pred_tree))
print(confusion_matrix(y_test, pred_tree))
print(classification_report(y_test, pred_tree))


# ===== NEURAL NETWORK =====
nn = Pipeline([
    ("pre", pre),
    ("scale", StandardScaler()),
    ("clf", MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        learning_rate_init=5e-4,
        batch_size=64,
        max_iter=800,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15
    ))
])

nn.fit(X_train, y_train)

pred_nn = nn.predict(X_test)

print("\n===== NEURAL NETWORK =====")
print("Accuracy:", accuracy_score(y_test, pred_nn))
print("Precision:", precision_score(y_test, pred_nn))
print("Recall:", recall_score(y_test, pred_nn))
print("F1:", f1_score(y_test, pred_nn))
print(confusion_matrix(y_test, pred_nn))
print(classification_report(y_test, pred_nn))