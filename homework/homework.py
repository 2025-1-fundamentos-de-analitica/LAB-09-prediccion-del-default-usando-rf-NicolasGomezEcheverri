# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

#Importar librerias
import pandas as pd
import os
import pickle
import json
import gzip
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import  precision_score, balanced_accuracy_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

def load_data():
    train = pd.read_csv(
        "../files/input/train_data.csv.zip",
        index_col=False,
        compression="zip",
    )
    test = pd.read_csv(
        "../files/input/test_data.csv.zip",
        index_col=False,
        compression="zip",
    )
    return train, test


# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
def clear_data(df):
    #Renombrar
    df.rename(columns={"default payment next month": "default"}, inplace=True)
    #Eliminacion de columna
    df = df.drop("ID", axis=1)
    #Eliminacion elementos nulos 
    df.dropna(inplace=True)
    #Cambia valores de educacion mayores a 4
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: x if x<=4 else 4)
    return df

# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
def make_train_test_split(df):
    #Division en etiquetas 
    y_df =  df["default"]
    #Division en caracteristicas de entrada
    x_df = df.drop("default", axis=1)
    return x_df, y_df

# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
def estimator_pipeline():
    # Define las columnas categóricas
    categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
    # Crea el preprocesador
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'  # Deja las columnas numéricas igual
    )
    #Contruccion pipeline
    pipeline = make_pipeline(
    preprocessor,
    RandomForestClassifier(),
    )
    return pipeline


# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
def cross_validation(estimator, x_train, y_train):
    #Hiperparametros a evaluar
    #Se debe antemoner el nombre del estimador y luego la lista de valores para un pipeline
    param_grid = {
        'randomforestclassifier__n_estimators': [75,120,175], #[100, 200],
        'randomforestclassifier__max_depth': [None, 5, 10], #| [None, 10, 20], #|
        'randomforestclassifier__min_samples_split': [2,5,10], #| [2, 5, 10], #|
        'randomforestclassifier__min_samples_leaf': [1,2] #| [1, 2, 4] #|
    }
    #Evaluacion de hiperparametros
    model = GridSearchCV(
        estimator= estimator,
        param_grid= param_grid,
        cv = 10,
        scoring="balanced_accuracy",
        refit=True,
        verbose=0,
        return_train_score=False,
    )
    #Aplicacion de GridSearchCV
    model.fit(x_train, y_train)
    #Informacion del mejor modelo y ademas definirlo
    print(model.best_score_)
    print(model.best_params_)
    return model

# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
def save_grid_search_model(model):
    #Guardar mejor modelo
    if not os.path.exists("../files/models"):
        os.makedirs("../files/models")
    with gzip.open("../files/models/model.pkl.gz", "wb") as file:
        pickle.dump(model, file)

# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}       
def eval_metrics(type_dataset, y_true, y_pred):

    b_accuracy = balanced_accuracy_score(y_true=y_true, y_pred=y_pred,)
    precision = precision_score(
        y_true=y_true,
        y_pred=y_pred, 
        labels=None, 
        pos_label=1,
        average="binary",)
    recall = recall_score(
        y_true=y_true,
        y_pred=y_pred, 
        labels=None, 
        pos_label=1,
        average="binary",)
    f1 = f1_score(
        y_true=y_true,
        y_pred=y_pred,
        labels=None,
        pos_label=1,
        average="binary",
        sample_weight=None,
        zero_division="warn",)

    #Formar diccionario de metricas 
    dic_metrics = { "type": "metrics",
                   'dataset': type_dataset, 
                   'precision': precision , 
                   'balanced_accuracy': b_accuracy, 
                   'recall': recall, 
                   'f1_score': f1}
    print(dic_metrics)
    #Guardar metricas como archivo json
    if not os.path.exists("../files/output"):
        os.makedirs("../files/output")
    with open("../files/output/metrics.json", "a") as f:
        json.dump(dic_metrics, f)
        f.write("\n")
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
def eval_confusion_matrix(type_dataset, y_true, y_pred):
    #         | Pronóstico
    #         |  PP    PN
    #---------|------------
    #      P  |  TP    FN
    # Real    |
    #      N  |  FP    TN

    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred,).ravel()

    #Formar diccionario de metricas 
    dic_confusion = {'type': 'cm_matrix', 'dataset': type_dataset, 
                   'true_0': {"predicted_0": int(tn), "predicte_1": int(fp)}, 
                   'true_1': {"predicted_0": int(fn), "predicted_1": int(tp)}}
    print(dic_confusion)
    #Guardar metricas como archivo json
    if not os.path.exists("../files/output"):
        os.makedirs("../files/output")
    with open("../files/output/metrics.json", "a") as f:
        json.dump(dic_confusion, f)
        f.write("\n")



#Carga de datos
train, test = load_data()
#Limpieza de datos
train = clear_data(train)
test = clear_data(test)
#Division en etiquetas y caracteristicas de entrada
x_train, y_train = make_train_test_split(train)
x_test, y_test = make_train_test_split(test)
#Creacion de pipelin para el empleo del estimador
estimator = estimator_pipeline()


#Entrenar y establecer mejor modelo deacuerdo a unos hiperparametros establecidos 
model = cross_validation(estimator, x_train, y_train)



#Salvar mejor model
save_grid_search_model(model)
# Calculo de métricas
eval_metrics("train", y_train, y_pred=model.best_estimator_.predict(x_train))
eval_metrics("test", y_test, y_pred=model.best_estimator_.predict(x_test))
# Calculo matriz de confusión
eval_confusion_matrix("train", y_train, y_pred=model.best_estimator_.predict(x_train))
eval_confusion_matrix("test", y_test, y_pred=model.best_estimator_.predict(x_test))