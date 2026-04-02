import pandas as pd

# ==============================
# LOAD DATA
# ==============================

file_path = 'data/release_10_23_2020.csv'
chunk_size = 500000

chunks = pd.read_csv(file_path, chunksize=chunk_size)

agg_list = []

for chunk in chunks:

    chunk['is_purchase'] = chunk['product_action'].apply(
        lambda x: 1 if x == 'purchase' else 0
    )

    temp = chunk.groupby('session_id_hash').agg({
        'event_type': 'count',
        'product_action': [
            lambda x: (x == 'detail').sum(),
            lambda x: (x == 'add').sum(),
            lambda x: (x == 'remove').sum(),
            lambda x: (x == 'purchase').sum()
        ],
        'server_timestamp_epoch_ms': ['min', 'max']
    })

    agg_list.append(temp)

# juntar tudo
session_df = pd.concat(agg_list)

# consolidar sessões repetidas entre chunks
session_df = session_df.groupby(level=0).sum()

df = pd.read_csv(file_path)

# ==============================
# VISÃO GERAL
# ==============================

print("\nHEAD:")
print(df.head())

print("\nINFO:")
df.info()

# ==============================
# DISTRIBUIÇÃO DE EVENTOS
# ==============================

print("\nEVENT TYPE:")
print(df['event_type'].value_counts())

# ==============================
# AÇÃO DO PRODUTO (CRÍTICO)
# ==============================

print("\nPRODUCT ACTION:")
print(df['product_action'].value_counts())

# ==============================
# VALORES NULOS
# ==============================

print("\nMISSING VALUES:")
print(df.isnull().sum())

# ==============================
# TARGET: COMPRA
# ==============================

df['is_purchase'] = df['product_action'].apply(lambda x: 1 if x == 'purchase' else 0)

print("\nPURCHASE DISTRIBUTION:")
print(df['is_purchase'].value_counts())

# ==============================
# AGRUPAMENTO POR SESSÃO
# ==============================

session_df = df.groupby('session_id_hash').agg({
    'event_type': 'count',
    'product_action': [
        lambda x: (x == 'detail').sum(),
        lambda x: (x == 'add').sum(),
        lambda x: (x == 'remove').sum(),
        lambda x: (x == 'purchase').sum()
    ],
    'server_timestamp_epoch_ms': ['min', 'max']
})

session_df.columns = [
    'num_events',
    'num_detail',
    'num_add',
    'num_remove',
    'num_purchase',
    'start_time',
    'end_time'
]

session_df = session_df.reset_index()

# ==============================
# FEATURE ENGINEERING
# ==============================

session_df['session_duration'] = session_df['end_time'] - session_df['start_time']
session_df['made_purchase'] = session_df['num_purchase'].apply(lambda x: 1 if x > 0 else 0)

print("\nSESSION DATA:")
print(session_df.head())

print("\nTARGET FINAL:")
print(session_df['made_purchase'].value_counts())

# ==============================
# MACHINE LEARNING
# ==============================

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

# features
features = [
    'num_events',
    'num_detail',
    'num_add',
    'num_remove',
    'session_duration'
]

X = session_df[features]
y = session_df['made_purchase']

# divisão
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# SMOTE (BALANCEAMENTO)
# ==============================

# aplicar SMOTE SOMENTE no treino
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# ==============================
# MODELO
# ==============================

# Substitui o modelo abaixo pelo atual
# model = RandomForestClassifier(
#     random_state=42,
#     class_weight='balanced'
# )
# model.fit(X_train, y_train)

model = RandomForestClassifier(random_state=42)
model.fit(X_train_res, y_train_res)

# previsão
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# avaliação
print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

print("\nROC AUC:")
print(roc_auc_score(y_test, y_proba))

# ==============================
# FEATURE IMPORTANCE
# ==============================

print("\nFEATURE IMPORTANCE:")

importance = model.feature_importances_

feat_imp = pd.DataFrame({
    'feature': features,
    'importance': importance
}).sort_values(by='importance', ascending=False)

#print(feat_imp)

#print("\nFEATURE IMPORTANCE (TOP):")
#print(feat_imp.to_string(index=False))

# ==============================
# EXIBIR GRAFICO DA FEATURE
# ==============================

import matplotlib.pyplot as plt

plt.figure()
plt.barh(feat_imp['feature'], feat_imp['importance'])
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()

plt.savefig("feature_importance.png")
plt.show()

# ==============================
# EXIBIR O TAMANHO REAL DO DATASET
# ==============================

with open(file_path, 'r', encoding='utf-8') as f:
    total_linhas = sum(1 for line in f) - 1  # tira o cabeçalho

print("\nTOTAL REAL DE LINHAS:")
print(total_linhas)

