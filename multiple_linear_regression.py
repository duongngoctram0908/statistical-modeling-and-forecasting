import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --- 1. Load Data ---
data = pd.read_csv("final-project-30firms.csv")
data_sub = data[["Y", "X1", "X2", "X3", "X4", "X5", "X6"]]

# --- 2. Scatter Matrix Plot ---
sns.pairplot(data_sub, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'})
plt.suptitle("Scatter Matrix Plot", y=1.02)
plt.show()

# --- 3. Full Model (Y ~ X1 + ... + X6) ---
model1 = smf.ols('Y ~ X1 + X2 + X3 + X4 + X5 + X6', data=data_sub).fit()
print(model1.summary())

# --- 4. Best Model (Y ~ X5 + X6) ---
model2 = smf.ols('Y ~ X5 + X6', data=data_sub).fit()
print(model2.summary())

# --- 5. Multicollinearity (VIF) ---
# Cần tạo matrix X (design matrix) cho model 1 để tính VIF
X_variables = data_sub[["X1", "X2", "X3", "X4", "X5", "X6"]]
X_variables = sm.add_constant(X_variables)
vif_data = pd.DataFrame()
vif_data["Feature"] = X_variables.columns
vif_data["VIF"] = [variance_inflation_factor(X_variables.values, i) for i in range(X_variables.shape[1])]
print("\n--- VIF Values ---")
print(vif_data)

# --- 6. Residual Analysis ---
# Standardized Residuals
influence = model2.get_influence()
std_resid = influence.resid_studentized_internal
y_hat = model2.fittedvalues

# QQ-Plot
fig, ax = plt.subplots(figsize=(6, 4))
sm.qqplot(std_resid, line='45', ax=ax, fit=True)
plt.title("QQ Plot of Residuals")
plt.show()

# Residuals vs Predicted Plot
plt.figure(figsize=(8, 5))
plt.scatter(y_hat, std_resid, color='green', alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Response")
plt.ylabel("Standardized Residuals")
plt.title("Residuals vs Fitted")
plt.show()

# --- 7. Cook's Distance ---
cooks_d = influence.cooks_distance[0]
plt.figure(figsize=(8, 5))
plt.stem(np.arange(len(cooks_d)), cooks_d, markerfmt=",")
plt.title("Cook's Distance")
plt.xlabel("Observation Index")
plt.ylabel("Cook's Distance")
plt.show()