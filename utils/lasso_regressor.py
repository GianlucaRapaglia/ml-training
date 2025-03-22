# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold

# Sample data
np.random.seed(42)
X = np.random.rand(100, 5)
y = X[:, 0] * 10 + np.random.randn(100)  # Generating a target with some noise

# Initialize Lasso model
lasso = Lasso(alpha=0.1)

# K-Fold setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(y))  # Placeholder for OOF predictions

# Perform cross-validation
for train_idx, val_idx in kf.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Fit LASSO model on training fold
    lasso.fit(X_train, y_train)
    
    # Store OOF predictions
    oof_preds[val_idx] = lasso.predict(X_val)

# Evaluate performance
from sklearn.metrics import mean_squared_error
oof_mse = mean_squared_error(y, oof_preds)
print("Out-of-Fold MSE:", oof_mse)

# %%
