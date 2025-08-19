import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb

# 1. Charger le CSV
df = pd.read_csv("metricExtraction/pet_features.csv")

# 2. Séparer X (features) et y (labels)
X = df.drop(columns=["label"])
y = df["label"]

# 3. Split train/test stratifié (80/20 par ex.)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4a. Random Forest baseline
rf = RandomForestClassifier(
    n_estimators=300, 
    max_depth=None, 
    class_weight="balanced", 
    random_state=42
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("=== Random Forest Report ===")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# 4b. XGBoost (souvent encore meilleur)
xgb_clf = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),  # gestion déséquilibre
    random_state=42,
    n_jobs=-1
)

xgb_clf.fit(X_train, y_train)
y_pred_xgb = xgb_clf.predict(X_test)

print("=== XGBoost Report ===")
print(classification_report(y_test, y_pred_xgb))
print(confusion_matrix(y_test, y_pred_xgb))

# 5. Cross-validation stratifiée (5 folds)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(rf, X, y, cv=cv, scoring="f1_macro")
print(f"CV F1-macro RF: {scores.mean():.3f} ± {scores.std():.3f}")
