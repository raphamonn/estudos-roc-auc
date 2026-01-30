# %%

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.metrics import (confusion_matrix,
                             classification_report,
                             roc_auc_score,
                             roc_curve)
import seaborn as sns
# %%
df = pd.read_csv('../data/diabetes.csv')


# %%
df.head()
# %%
df.isnull().sum()
# %%
sns.pairplot(df, hue='Outcome')
# %%
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
scale = MinMaxScaler()
# %%
X_train_scaled = scale.fit_transform(X_train)
X_test_scaled = scale.transform(X_test)
# %%
# TREINAMENTOS
# Versão sem scaling
knn = KNeighborsClassifier(metric='euclidean',
                           n_neighbors=4,
                           random_state=42),
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
knn_predict_proba = knn.predict_proba(X_test)[:, 1]

# %%
# Versão Com scaling
knn_v2 = KNeighborsClassifier(
    metric='euclidean',
    n_neighbors=4,
    random_state=42)
knn_v2.fit(X_train_scaled, y_train)
y_pred_v2 = knn_v2.predict(X_test_scaled)
knn_predict_proba_v2 = knn_v2.predict_proba(X_test_scaled)[:, 1]
# %%
# Random Forest (Ele não necessita de Scaling)
rf = RandomForestClassifier(
    n_estimators=200,
    criterion='gini',
    random_state=42,
    max_depth=5,
    max_features='sqrt'
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_predict_proba = rf.predict_proba(X_test)[:, 1]
# %%
# Decision Tree
dt = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)
y_pred_dt = dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
dt_predict_proba = dt.predict_proba(X_test)[:, 1]
# %%
# KNN SEM SCALING
print('KNN Versão sem Scaling')
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# KNN COM SCALING
print('KNN Versão com Scaling')
print(classification_report(y_test, y_pred_v2))
print(confusion_matrix(y_test, y_pred_v2))
# RANDOM FOREST
print('Random Forest')
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
# %%
# KNN sem scaling
fpr_knn_1, tpr_knn_1, thresholds = roc_curve(
    y_true=y_test, y_score=knn_predict_proba)

# Knn com Scaling
fpr_knn_2, tpr_knn_2, thresholds = roc_curve(
    y_true=y_test, y_score=knn_predict_proba_v2)

# RF
fpr_rf, tpr_rf, thresholds = roc_curve(
    y_true=y_test, y_score=rf_predict_proba)

fpr_dt, tpr_dt, thresholds = roc_curve(
    y_true=y_test, y_score=dt_predict_proba)
# %%
auc_knn_1 = roc_auc_score(y_test, knn_predict_proba)
auc_knn_2 = roc_auc_score(y_test, knn_predict_proba_v2)
auc_rf = roc_auc_score(y_test, rf_predict_proba)
auc_dt = roc_auc_score(y_test, dt_predict_proba)

# %%
plt.plot(
    fpr_knn_1,
    fpr_knn_1,
    linestyle='--',
    color='grey')

plt.plot(
    fpr_knn_2,
    tpr_knn_2,
    color='red',
    label=f"KNN com scaling - AUC {auc_knn_1:.2f}")

plt.plot(fpr_knn_1, tpr_knn_1, color='green',
         label=f'KNN sem scaling - AUC {auc_knn_2:.2f}')

plt.plot(
    fpr_dt,
    tpr_dt,
    color='magenta',
    label=f'Decision Tree - AUC {auc_rf:.2f}')

plt.plot(
    fpr_rf,
    tpr_rf,
    color='blue',
    label=f'Random Forest - AUC {auc_dt:.2f}')

plt.title('Curva ROC')
plt.ylabel('TPR - (True Positive Rate)')
plt.xlabel('FPR - (False Positive Rate)')
plt.legend(loc='lower right')
plt.show()
# %%
