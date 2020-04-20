"""
Prediccion de acciones en la bolsa usando redes LSTM
"""

import numpy as np
np.random.seed(4)
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Funcion auxiliar para graficar
def graficar_predicciones(real, prediccion):
    plt.plot(real[0:len(prediccion)], color='red', label='Valor real de la accion')
    plt.plot(prediccion, color='blue', label='Prediccion de la accion')
    plt.ylim(1.1*np.min(prediccion)/2, 1.1*np.max(prediccion))
    plt.xlabel('Tiempo')
    plt.ylabel('Valor de la accion')
    plt.legend()
    plt.show()

# Lectura de datos
dataset = pd.read_csv("AAPL_2006-01-01_to_2018-01-01.csv", index_col='Date', parse_dates=['Date'])
dataset.head()

# Sets de entrenamiento y validacion
# La LSTM se entrenara con datos de 2016 hacia atras, la validacion se hara con datos del 2017 en
# adelante. Se usara solo el valor mas alto de la accion para cada dia (columna High)
set_entrenamiento = dataset[:'2016'].iloc[:,1:2]
set_validacion = dataset['2017':].iloc[:,1:2]

set_entrenamiento['High'].plot(legend=True)
set_validacion['High'].plot(legend=True)
plt.legend(['Entrenamiento (2006-2016', 'Validacion (2017)'])
plt.show()

# Normalizacion del set de entrenamiento
sc = MinMaxScaler(feature_range=(0,1))
set_entrenamiento_escalado = sc.fit_transform(set_entrenamiento)

# La red LSTM tendra como entrada "time_step" datos consecutivos y como salida 1 dato
# el cual es la prediccion a partir de la estampa de tiempo dada. Se conformara de esta
# forma el set de entrenamiento
time_step = 60
X_train = []    # (2708, 60)
Y_train = []    # (2708,)
m = len(set_entrenamiento_escalado)

for i in range(time_step, m):
    # X: bloques de "time_step" datos: 0-time_step, 1-time_step+1, 2-time_step+2, etc
    X_train.append(set_entrenamiento_escalado[i-time_step:i,0])

    # Y: el siguiente dato
    Y_train.append(set_entrenamiento_escalado[i,0])
X_train, Y_train = np.array(X_train), np.array(Y_train)

# Reshape X_train para que se ajuste al modelo en Keras (COnvertir de matriz a vector
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  # (2708, 60, 1)

# Construccion red LSTM
dim_entrada = (X_train.shape[1],1)
dim_salida = 1
na = 50     # Cantidad de neuronas

# Creacion el modelo
modelo = Sequential()
modelo.add(LSTM(units=na, input_shape=dim_entrada)) # Red LSTM con 50 neuronas y dimension de entrada de 60
modelo.add(Dense(units=dim_salida))
modelo.compile(optimizer='rmsprop', loss='mse')     # El optimizador rmsprop funciona similar al Gradiente Descendente
modelo.fit(X_train,Y_train,epochs=20,batch_size=32)     # Entrenamos el modelo con 20 iteraciones y lotes de 32 datos por iteracion

# Validacion (prediccion del valor de las acciones)
x_test = set_validacion.values
x_test = sc.transform(x_test)

X_test = []
for i in range(time_step, len(x_test)):
    X_test.append(x_test[i-time_step:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))

prediccion = modelo.predict(X_test)
prediccion = sc.inverse_transform(prediccion)   # Aplicamos tranformacion inversa a los datos normalizados]

# Graficar resultados
graficar_predicciones(set_validacion.values, prediccion)
