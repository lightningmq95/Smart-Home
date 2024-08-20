import xgboost as xgb
from xgboost import XGBRFClassifier, XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import h5py
import numpy as np

with h5py.File('./fist_gesture_features.h5', 'r') as f:
    X_onoff = f['features'][:]
    y_onoff = f['labels'][:]

X_train_onoff, X_test_onoff, y_train_onoff, y_test_onoff = train_test_split(X_onoff, y_onoff, test_size=0.2, random_state=38)

xgb_rf_onoff = XGBRFClassifier(
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

xgb_gb_onoff = XGBClassifier(
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

xgb_ext_onoff = XGBClassifier(
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

estimators_onoff = [
    ('rf', xgb_rf_onoff),
    ('gb', xgb_gb_onoff),
    ('ext', xgb_ext_onoff)
]

stacked_model_onoff = StackingClassifier(
    estimators=estimators_onoff,
    final_estimator=xgb_ext_onoff,
    n_jobs=-1
)

cv_scores_onoff = cross_val_score(stacked_model_onoff, X_train_onoff, y_train_onoff, cv=5, scoring='accuracy')
print(f"Cross-validation Accuracy Scores (On-Off): {cv_scores_onoff}")
print(f"Mean CV Accuracy (On-Off): {cv_scores_onoff.mean()}")

stacked_model_onoff.fit(X_train_onoff, y_train_onoff)
y_pred_onoff = stacked_model_onoff.predict(X_test_onoff)

print(f"Accuracy: {accuracy_score(y_test_onoff, y_pred_onoff)}")
print(f"Precision: {precision_score(y_test_onoff, y_pred_onoff)}")
print(f"Recall: {recall_score(y_test_onoff, y_pred_onoff)}")
print(f"F1 Score: {f1_score(y_test_onoff, y_pred_onoff)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test_onoff, y_pred_onoff)}")

with open('onoff_gesture_model.pkl', 'wb') as f:
    pickle.dump(stacked_model_onoff, f)
