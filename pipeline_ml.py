# ================================
# Importações e Configuração de Logs
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, matthews_corrcoef
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import KNNImputer
import joblib

logging.basicConfig(
    filename='pipeline_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ================================
# Funções de cada etapa
# ================================

def carregar_dados():
    logging.info("Carregando dados...")
    train = pd.read_csv('bootcamp_train.csv')
    test = pd.read_csv('bootcamp_test.csv')
    return train.copy(), test.copy()

def tratar_valores_ausentes(train, test):
    logging.info("Tratando valores ausentes...")
    numerical_cols = train.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = train.select_dtypes(include=['object']).columns

    knn_imputer = KNNImputer()
    train[numerical_cols] = knn_imputer.fit_transform(train[numerical_cols])
    test[numerical_cols] = knn_imputer.transform(test[numerical_cols])

    for col in categorical_cols:
        train[col].fillna(train[col].mode()[0], inplace=True)
        if col in test.columns:
            test[col].fillna(train[col].mode()[0], inplace=True)

    for col in ['falha_1', 'falha_2', 'falha_3', 'falha_4', 'falha_5', 'falha_6']:
        train[col] = train[col].astype(str).str.lower().isin(['true', 'sim', 'yes', '1'])
        if col in test.columns:
            test[col] = test[col].astype(str).str.lower().isin(['true', 'sim', 'yes', '1'])
    return train, test

def tratar_valores_negativos_e_normalizar(train, test):
    logging.info("Tratando valores negativos e normalizando...")
    columns_positive_only = ['area_pixels', 'perimetro_x', 'perimetro_y',
                             'comprimento_do_transportador', 'espessura_da_chapa_de_aço']
    for col in columns_positive_only:
        if col in train.columns:
            median_value = train[col][train[col] >= 0].median()
            train[col] = train[col].apply(lambda x: x if x >= 0 else median_value)
        if col in test.columns:
            test[col] = test[col].apply(lambda x: x if x >= 0 else median_value)

    columns_to_normalize = ['x_minimo', 'x_maximo', 'y_minimo', 'y_maximo',
                            'indice_de_variacao_x', 'indice_de_variacao_y',
                            'indice_de_orientação', 'indice_de_luminosidade']

    columns_to_normalize_existing = [col for col in columns_to_normalize if col in train.columns and col in test.columns]
    if columns_to_normalize_existing:
        scaler = MinMaxScaler()
        train[columns_to_normalize_existing] = scaler.fit_transform(train[columns_to_normalize_existing])
        test[columns_to_normalize_existing] = scaler.transform(test[columns_to_normalize_existing])
    return train, test

def criar_variavel_alvo(train, test):
    logging.info("Criando variável alvo...")

    def identificar_classe_falha(row):
        for i in range(1, 7):
            if row[f'falha_{i}']:
                return f'falha_{i}'
        return 'falha_outros'

    train['classe_defeito'] = train.apply(identificar_classe_falha, axis=1)

    if all(col in test.columns for col in ['falha_1', 'falha_2', 'falha_3', 'falha_4', 'falha_5', 'falha_6']):
        test['classe_defeito'] = test.apply(identificar_classe_falha, axis=1)

    class_mapping = {label: idx for idx, label in enumerate(train['classe_defeito'].unique())}
    train['classe_defeito'] = train['classe_defeito'].map(class_mapping)

    if 'classe_defeito' in test.columns:
        test['classe_defeito'] = test['classe_defeito'].map(class_mapping)

    return train, test, class_mapping

def preparar_dados(train, test):
    logging.info("Preparando dados para treinamento...")
    X = train.drop(columns=['classe_defeito', 'falha_1', 'falha_2', 'falha_3', 'falha_4', 'falha_5', 'falha_6'], errors='ignore')
    y = train['classe_defeito']

    X = pd.get_dummies(X, drop_first=True)
    X_test_final = pd.get_dummies(test.drop(columns=['classe_defeito', 'falha_1', 'falha_2', 'falha_3', 'falha_4', 'falha_5', 'falha_6'], errors='ignore'),
                                   drop_first=True)

    missing_cols = set(X.columns) - set(X_test_final.columns)
    for col in missing_cols:
        X_test_final[col] = 0
    X_test_final = X_test_final[X.columns]

    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test_final)

    smote_tomek = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smote_tomek.fit_resample(X_train_scaled, y_train)

    return X_resampled, y_resampled, X_val_scaled, y_val, X_test_scaled, scaler

def treinar_modelos(X_resampled, y_resampled, X_val_scaled, y_val, class_mapping):
    logging.info("Treinando modelos...")
    modelos = {
        'Random Forest': (RandomForestClassifier(random_state=42), {'n_estimators': [100, 200], 'max_depth': [10, 20]}),
        'Regressão Logística': (LogisticRegression(max_iter=1000, class_weight='balanced'), {'C': [0.1, 1, 10]}),
        'SVM': (SVC(probability=True, class_weight='balanced'), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),
        'XGBoost': (XGBClassifier(eval_metric='mlogloss', random_state=42),
                    {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 6]}),
        'LightGBM': (LGBMClassifier(random_state=42),
                     {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 6]}),
        'MLP': (MLPClassifier(random_state=42), {'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.0001, 0.001]})
    }

    melhores_modelos = {}
    for nome, (modelo, params) in modelos.items():
        logging.info(f"Tunando {nome}...")
        grid = GridSearchCV(modelo, params, cv=StratifiedKFold(n_splits=5), scoring='accuracy', n_jobs=-1)
        grid.fit(X_resampled, y_resampled)
        melhores_modelos[nome] = grid.best_estimator_
    return melhores_modelos

def salvar_modelo_e_previsoes(melhores_modelos, X_test_scaled, test, class_mapping):
    melhor_modelo_nome = max(melhores_modelos, key=lambda nome: melhores_modelos[nome].score(X_test_scaled, test['classe_defeito']))
    melhor_modelo = melhores_modelos[melhor_modelo_nome]

    joblib.dump(melhor_modelo, f"melhor_modelo_{melhor_modelo_nome}.pkl")
    logging.info(f"Melhor modelo salvo: {melhor_modelo_nome}")

    y_test_pred = melhor_modelo.predict(X_test_scaled)
    test['classe_predita'] = y_test_pred
    test['classe_predita'] = test['classe_predita'].map({v: k for k, v in class_mapping.items()})
    test.to_csv('bootcamp_test_predictions.csv', index=False)

# ================================
# Execução Principal
# ================================

def main():
    train, test = carregar_dados()
    train, test = tratar_valores_ausentes(train, test)
    train, test = tratar_valores_negativos_e_normalizar(train, test)
    train, test, class_mapping = criar_variavel_alvo(train, test)
    X_resampled, y_resampled, X_val_scaled, y_val, X_test_scaled, scaler = preparar_dados(train, test)
    melhores_modelos = treinar_modelos(X_resampled, y_resampled, X_val_scaled, y_val, class_mapping)
    salvar_modelo_e_previsoes(melhores_modelos, X_test_scaled, test, class_mapping)

if __name__ == "__main__":
    main()