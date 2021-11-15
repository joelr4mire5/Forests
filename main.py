import os
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from predictPy import Analisis_Predictivo
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns

print("Tarea 3")

print("Pregunta #1 Carga de los datos ")
potability_df= pd.read_csv("Input/potabilidad_V2.csv")
potability_columns=potability_df.columns
potability_df=potability_df[potability_columns[1:]]


instancia_tree = DecisionTreeClassifier(min_samples_split = 2, max_depth=None,
                                        criterion="gini")

instancia_ada = AdaBoostClassifier(base_estimator=instancia_tree,
                                            n_estimators=200)


analisis_potability_ada_boosting = Analisis_Predictivo(potability_df, predecir="Potability", modelo=instancia_ada, estandarizar= True,
                                                       train_size= 0.75)
resultados_potabilidad_ada_boosting= analisis_potability_ada_boosting.fit_predict_resultados()
print("Resultados Ada Boosting Pregunta 1.a")
resultados_potabilidad_ada_boosting

instancia_gbc = GradientBoostingClassifier(n_estimators=100,min_samples_split=2, random_state=0)

analisis_potability_gbc = Analisis_Predictivo(potability_df,predecir= "Potability",modelo=instancia_gbc,train_size= 0.75, random_state = 0)
resultados_potabilidad_gbc= analisis_potability_gbc.fit_predict_resultados()


print("Resultados GBC Pregunta 1.a")
resultados_potabilidad_gbc

print("Resultados Ada Boosting Pregunta 1.b")
print("Importancia de de variables Ada boosting")
#Obtenemos valores
importancia = np.array(analisis_potability_ada_boosting.modelo.feature_importances_)
etiquetas = np.array(analisis_potability_ada_boosting.predictoras)

#Ordenar
orden = np.argsort(importancia)
importancia = importancia[orden]
etiquetas = etiquetas[orden]

print("Importancia de las variables: ",importancia, "\n")
print(etiquetas)

#Gráfico
fig, ax = plt.subplots(1,1, figsize = (12,6), dpi = 200)
ax.barh(etiquetas, importancia)
plt.show()

print("Resultados GBC  Pregunta 1.b")

print("Importancia de de variables GBC")
#Obtenemos valores
importancia = np.array(analisis_potability_gbc.modelo.feature_importances_)
etiquetas = np.array(analisis_potability_gbc.predictoras)

#Ordenar
orden = np.argsort(importancia)
importancia = importancia[orden]
etiquetas = etiquetas[orden]

print("Importancia de las variables: ",importancia, "\n")
print(etiquetas)

#Gráfico
fig, ax = plt.subplots(1,1, figsize = (12,6), dpi = 200)
ax.barh(etiquetas, importancia)
plt.show()