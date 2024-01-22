#Bibliotecas necesarias
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sb

# Descargo el CSV
csv_file_url = 'https://drive.google.com/file/d/1reUBOFFwRdL9s5SMaYMC5wVnCaWnAH-k/view?usp=sharing'
file_id = csv_file_url.split('/')[-2]
dwn_url = 'https://drive.google.com/uc?export=download&id=' + file_id
df = pd.read_csv(dwn_url, sep=';')
df = df.set_index(['user_uuid', 'course_uuid', 'particion'])

# Defino las variables independientes (predictores)
X = df[['nota_parcial', 'score']] 

# Defino la variable dependiente (nota_final_materia)
Y = df['nota_final_materia']

# Creo un imputador que rellena los valores NaN con la mediana de la columna
imputer = SimpleImputer(strategy='median')

# Divido el conjunto de datos en entrenamiento y evaluación
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Guardo los índices de X_test antes de la transformación
index_X_test = X_test.index

# Aplico el imputador a las variables independientes
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Ajusto el modelo de regresión logística
model = LogisticRegression(max_iter=10000)
model.fit(X_train_imputed, y_train)

# Obtengo la probabilidad de desprobar para cada observación del conjunto de evaluación
y_proba = model.predict_proba(X_test_imputed)
y_proba_desprobar = y_proba[:, 1]

# Crea Datos_Final sin las columnas 'user_uuid' y 'course_uuid'
Datos_Final = pd.DataFrame({'probabilidad': y_proba_desprobar}, index=index_X_test)

# Filto el dataframe
df_filtrado = Datos_Final.loc[Datos_Final['probabilidad'] > 0.2]
print('--Muestra Previa--',df_filtrado)
# Muestro la probabilidad en forma de porcentaje
df_filtrado.loc[:, 'porcentaje'] = df_filtrado['probabilidad'].apply(lambda x: round(x*100, 2))

# Agrupo por 'user_uuid' y 'course_uuid' y calculo el promedio del porcentaje
df_agrupado = df_filtrado.groupby(['user_uuid', 'course_uuid'])['porcentaje'].mean()

# Muestro el dataframe agrupado
print('-----------------------------------Probabilidad de desaprobar---------------------------------------------\n',df_agrupado,'%')
