# %%

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, auc, roc_curve
import seaborn as sns
# %%
df = pd.read_csv('../datasets/diabetes.csv')


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
# Versão sem scaling
# Quero comparar o ganho de AUC quando eu uso o
knn = KNeighborsClassifier(metric='euclidean', n_neighbors=4)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
knn_predict_proba = knn.predict_proba(X_test)[:, 1]
# %%
# Versão Com scaling
knn_v2 = KNeighborsClassifier(metric='euclidean', n_neighbors=4)
knn_v2.fit(X_train_scaled, y_train)
y_pred_v2 = knn_v2.predict(X_test_scaled)
knn_predict_proba_v2 = knn_v2.predict_proba(X_test_scaled)[:, 1]
# %%
print('Versão sem Scaling')
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# %%
# versão com scaling
print('Versão com Scaling')
print(classification_report(y_test, y_pred_v2))
print(confusion_matrix(y_test, y_pred_v2))
# %%
fpr, tpr, thresholds = roc_curve(
    y_true=y_test, y_score=knn_predict_proba, thresholds=[0.1, 0.])
# %%

plt.set_tile
plt.plot(fpr, tpr)
# %%
# %%
