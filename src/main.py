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
                             accuracy_score,
                             roc_auc_score,
                             roc_curve)
import seaborn as sns
from sklearn.pipeline import Pipeline
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
# %%
# TREINAMENTO USANDO PIPELINES
models = {
    'KNN_sem_scaling': Pipeline(steps=[
        ("model", KNeighborsClassifier(n_neighbors=4, metric='euclidean'))
    ]),
    'KNN_com_scaling': Pipeline(steps=[
        ("scaling", MinMaxScaler()),
        ("model", KNeighborsClassifier(n_neighbors=4, metric='euclidean'))
    ]),
    'RandomForest': Pipeline(steps=[
        ('model', RandomForestClassifier(
            n_estimators=200,
            criterion='gini',
            max_depth=5,
            max_features='sqrt',
            random_state=42)
         ),]),
    'DecisionTree': Pipeline(steps=[
        ('model', DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ))
    ])
}
# %%
results = {}

for name, model in models.items():
    model.fit(X_train, y_train),

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "model": model,
        "y_pred": y_pred,
        "y_proba": y_proba
    }


# %%
for model, info in results.items():
    print(f"Classification report do {model}")
    print(classification_report(y_test, info['y_pred']))
# %%

for model, info in results.items():
    print(f'Matriz de confusão do {model}')
    print(confusion_matrix(y_test, info['y_pred']))
    print(f'\n acurácia: {accuracy_score(y_test, info['y_pred']):.2f}')

# %%
for model, info in results.items():
    (results[model]['fpr'],
     results[model]['tpr'],
     thresholds) = roc_curve(y_test, info['y_proba'])

    results[model]['auc_score'] = roc_auc_score(y_test, info['y_proba'])
# %%
# PLOTANDO OS RESULTADOS
plt.plot([0, 1], [0, 1],
         linestyle='--',
         color='grey')

for model, info in results.items():

    plt.plot(
        info['fpr'],
        info['tpr'],
        label=f"{model} - AUC {info['auc_score']:.2f}",
    )

plt.title('Curva ROC')
plt.ylabel('TPR - (True Positive Rate)')
plt.xlabel('FPR - (False Positive Rate)')
plt.legend(loc='lower right')
plt.show()
# %%
