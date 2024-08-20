import xgboost as xgb
import numpy as np
import h5py

with h5py.File('./Data/combined_gesture_features.h5', 'r') as f:
    print("Datasets in the file:")
    print(list(f.keys()))

    combined_features = f['combined_features']
    print("Keys in 'combined_features':")
    print(list(combined_features.keys()))

    X = combined_features['X'][:]
    y = combined_features['y'][:]

dtrain = xgb.DMatrix(X, label=y)

param_rf = {
    'max_depth': 100,
    'eta': 0.1,
    'nthread': 4,
    'num_parallel_tree': 1000,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'binary:logistic',
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor'
}

param_gb = {
    'max_depth': 50,
    'eta': 0.1,
    'nthread': 25,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor'
}

param_eb = {
    'max_depth': 60,
    'eta': 0.05,
    'nthread': 25,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor'
}

num_round_rf = 100
bst_rf = xgb.train(param_rf, dtrain, num_round_rf)

num_round_gb = 100
bst_gb = xgb.train(param_gb, dtrain, num_round_gb)

num_round_eb = 100
bst_eb = xgb.train(param_eb, dtrain, num_round_eb)

X_stack = np.column_stack([
    bst_rf.predict(dtrain),
    bst_gb.predict(dtrain),
    bst_eb.predict(dtrain)
])

dtrain_stack = xgb.DMatrix(X_stack, label=y)

param_stack = {
    'max_depth': 6,
    'eta': 0.1,
    'nthread': 4,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor'
}

num_round_stack = 100
bst_stack = xgb.train(param_stack, dtrain_stack, num_round_stack)

bst_stack.save_model('stacked_model.bin')

def evaluate_model(preds, labels):
    preds_binary = (preds > 0.5).astype(int)
    accuracy = np.mean(preds_binary == labels)
    precision = np.sum((preds_binary == 1) & (labels == 1)) / np.sum(preds_binary == 1)
    recall = np.sum((preds_binary == 1) & (labels == 1)) / np.sum(labels == 1)
    f1_score = 2 * (precision * recall) / (precision + recall)

    tp = np.sum((preds_binary == 1) & (labels == 1))
    tn = np.sum((preds_binary == 0) & (labels == 0))
    fp = np.sum((preds_binary == 1) & (labels == 0))
    fn = np.sum((preds_binary == 0) & (labels == 1))

    confusion_matrix = np.array([[tn, fp], [fn, tp]])

    return accuracy, precision, recall, f1_score, confusion_matrix

print("Random Forest Metrics:")
preds_rf = bst_rf.predict(dtrain)
accuracy_rf, precision_rf, recall_rf, f1_rf, conf_matrix_rf = evaluate_model(preds_rf, y)
print(f'Accuracy: {accuracy_rf:.4f}, Precision: {precision_rf:.4f}, Recall: {recall_rf:.4f}, F1 Score: {f1_rf:.4f}')
print(f'Confusion Matrix:\n{conf_matrix_rf}')

print("\nGradient Boosting Metrics:")
preds_gb = bst_gb.predict(dtrain)
accuracy_gb, precision_gb, recall_gb, f1_gb, conf_matrix_gb = evaluate_model(preds_gb, y)
print(f'Accuracy: {accuracy_gb:.4f}, Precision: {precision_gb:.4f}, Recall: {recall_gb:.4f}, F1 Score: {f1_gb:.4f}')
print(f'Confusion Matrix:\n{conf_matrix_gb}')

print("\nExtreme Boosting Metrics:")
preds_eb = bst_eb.predict(dtrain)
accuracy_eb, precision_eb, recall_eb, f1_eb, conf_matrix_eb = evaluate_model(preds_eb, y)
print(f'Accuracy: {accuracy_eb:.4f}, Precision: {precision_eb:.4f}, Recall: {recall_eb:.4f}, F1 Score: {f1_eb:.4f}')
print(f'Confusion Matrix:\n{conf_matrix_eb}')

print("\nStacked Model Metrics:")
preds_stack = bst_stack.predict(dtrain_stack)
accuracy_stack, precision_stack, recall_stack, f1_stack, conf_matrix_stack = evaluate_model(preds_stack, y)
print(f'Accuracy: {accuracy_stack:.4f}, Precision: {precision_stack:.4f}, Recall: {recall_stack:.4f}, F1 Score: {f1_stack:.4f}')
print(f'Confusion Matrix:\n{conf_matrix_stack}')

print("Model training complete and stacked model saved.")
