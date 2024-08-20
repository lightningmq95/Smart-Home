import xgboost as xgb
from xgboost import XGBRFClassifier, XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import h5py
import numpy as np

with h5py.File('./finger_circle_features.h5', 'r') as f:
    X_circle = f['features'][:]
    y_circle = f['labels'][:]

X_train_circle, X_test_circle, y_train_circle, y_test_circle = train_test_split(X_circle, y_circle, test_size=0.2, random_state=38)

xgb_rf_circle = XGBRFClassifier(
    colsample_bynode=0.8,
    learning_rate=1,
    max_depth=5,
    num_parallel_tree=500,
    objective='binary:logistic',
    subsample=0.8,
    tree_method='hist',
    device='cuda',
    n_estimators=1000
)

xgb_gb_circle = XGBClassifier(
    tree_method='hist',
    device='cuda',
    n_estimators=1000,
    max_depth=15,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    booster='gbtree'
)

xgb_ext_circle = XGBClassifier(
    tree_method='hist',
    device='cuda',
    n_estimators=1000,
    max_depth=15,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    booster='gbtree'
)

estimators_circle = [
    ('rf', xgb_rf_circle),
    ('gb', xgb_gb_circle),
    ('ext', xgb_ext_circle)
]

stacked_model_circle = StackingClassifier(
    estimators=estimators_circle,
    final_estimator=xgb_ext_circle,
    n_jobs=-1
)

cv_scores_circle = cross_val_score(stacked_model_circle, X_train_circle, y_train_circle, cv=5, scoring='accuracy')
print(f"Cross-validation Accuracy Scores (Circle Gesture): {cv_scores_circle}")
print(f"Mean CV Accuracy (Circle Gesture): {cv_scores_circle.mean()}")

stacked_model_circle.fit(X_train_circle, y_train_circle)
y_pred_circle = stacked_model_circle.predict(X_test_circle)

print(f"Circle Gesture Model Accuracy: {accuracy_score(y_test_circle, y_pred_circle)}")
print(f"Circle Gesture Model Precision: {precision_score(y_test_circle, y_pred_circle)}")
print(f"Circle Gesture Model Recall: {recall_score(y_test_circle, y_pred_circle)}")
print(f"Circle Gesture Model F1 Score: {f1_score(y_test_circle, y_pred_circle)}")
print(f"Circle Gesture Model Confusion Matrix:\n{confusion_matrix(y_test_circle, y_pred_circle)}")

with open('finger_circle_model.pkl', 'wb') as f:
    pickle.dump(stacked_model_circle, f)
