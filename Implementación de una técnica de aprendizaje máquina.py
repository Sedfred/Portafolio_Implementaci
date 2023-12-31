#importar librerias necesarias
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
#leer la data
df=pd.read_csv("C:/Users/Admin/Downloads/titanic.csv")
#mostrar la forma de los datos originales
print("Data sin limpieza:", df.shape)
#eliminar columnas que no son importantes para el analisis
df.drop(["PassengerId", "Name","Ticket","Cabin"], axis=1, inplace=True)
#encontrar los valores vacios
print("Valores vacios en la data:\n",df.isnull().sum())
#rellenar los valores vacios en la columna Age con la mediana de los datos
mage = df['Age'].median()
df['Age'].fillna(mage, inplace=True)
#eliminar el resto de valores vacios pues son pocos
df.dropna(inplace=True)
print("Valores vacios en la data despues de la limpeza:\n",df.isnull().sum())
#utilizar labelencoder para reemplazar los valores categoricos por numericos asignado valores apartir de 1 dependiendo de los valores unicos presentes
# en cada calumna
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['Embarked'] = label_encoder.fit_transform(df['Embarked'])
#dado que el modelo que deseamos uitlizar se ve altamente afectado por las grandes diferencias numericas que existen en nuestros datos
#utilizaremos MinMaxscaler para realizar una transformacion en los datos
scaler = MinMaxScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])
print("normalizacion de la data")
#finalmente reacomodaremos los datos
df=df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
       'Embarked','Survived']]


#Se decidio implementar knn pues es un algoritmo facil de aplicar y ademas 
#al tener solo dos clusters sera mas facil para el algoritmo determinar a que clase perteneceran las nuevas predicciones

#creamos la funcion que calculcara la distancia euclidiana entre dos puntos
def distancia_euc(p1, p2):
  """Calcula la distancia euclidiana entre dos puntos en un espacio n-dimensional
    p1: Coordenadas del primer punto
    p2: Coordenadas del segundo punto
   Devuelve la distancia euclidiana entre los puntos"""
  sum_disc = 0.0
  #en este ciclo for utilizamos zip para poder trabajar con las coordendas de los dos ditintos puntos al mismo tiempo
  for cord1, cord2 in zip(p1, p2):
    #se realiza la resta de la cordenada del punto 1 con la del punto 2 y se eleva al cuadrado y se realiza la suma de estos valores
    #durante el ciclo
    sum_disc += (cord1 - cord2) ** 2
#se calcula la raiz cuadral resultado de la suma
  distancia_euc = math.sqrt(sum_disc)
  #devuelve la distancia
  return distancia_euc

#ahora definimos la funcion que realiza el algoritmo knn
def knn(data, point, k):
  """Implementa el algoritmo k-Nearest Neighbors (k-NN) para predecir la clase de un punto
    data: Conjunto de datos de entrenamiento
    point: Punto a predecir
    k: Número de vecinos a considerar
   Devuelve la clase predicha para el punto"""
  dist = []
  #ireamos sobre la data e ignoramos el primer valor que nos devuelve iterrows()
  for _, i in data.iterrows():
      #calculamos la distancia entre el punto a predecir y los demas valores
      distancia = distancia_euc(point, i[:-1])
    #obtener el valor el cual no ultilizamos en la en 
      label = i[-1]
      #agregamos una tupla a la lista de distancias
      dist.append((distancia, label))
#ordenamos la lista de distancias  de menor a mayor
  dist.sort()
  #creamos un lista para almacar etiquetas de los k vecinos mas cercanos al punto
  label_kn = []
  #creamos un ciclo que nos devolvera los labels de los 6 puntos mas cercanos (6 por el valor de k) y
  #  excruira la distancia almacenada en la lista de tuplas
  for _, l in dist[:k]:
      label_kn.append(l)
#creamos una nueva lista 
  list_lable = {}
  #creamo un ciclo para iterar sobre la lista de puntos cercanos del punto anterior
  for label in label_kn:
      #este ciclo contara la veces que aparece cada punto los elementos cercanos
      if label in list_lable:
          list_lable[label] += 1
      else:
          list_lable[label] = 1
  #con la funcion max econtramos la etiqueta o valor en el que se clasifico con mayor frecuencia
  most_common_label = max(list_lable, key=list_lable.get)
  #regresamos este valor
  return most_common_label

k = 6
#creamos una lista que guardara las matrices de confusion de cada iteracion
list_cm=[]

#creamos un ciclo para realizar las pruebas correspondientes
#le damos valor a n la cual sera el control para el numero de pruebas a realizar y subplots
n=4
for i in range(n):
    #imprimimos un texto para diferenciar cada prueba
    print("++++++++PRUEBA ", i+1,"+++++++++")
    #con ayuda de las funciones de sklearn hacemos la division de la data en 3 conjuntos, uno de entrenmiento con el 60% de los datos, 
    #uno de validacion y test con el 20% de los datos respectivamente
    train_df, temp_df = train_test_split(df, test_size=0.4, )
    test_df, validation_df = train_test_split(temp_df, test_size=0.5, )
    #imprimimos el numero de datos que tiene cada conjunto de datos la primera vez
    if i==0:
        print("Tamaño del conjunto de entrenamiento:", len(train_df))
        print("Tamaño del conjunto de prueba:", len(test_df))
        print("Tamaño del conjunto de validación:", len(validation_df))
    #asignamos las variables X y Y para todos los conjuntos de prueba y validacion
    #Para X tomamos todos las columnas menos el ultimo valor, y para y tomames esta columna faltante
    X_test=test_df.iloc[:,:-1]
    y_test=test_df["Survived"]
    X_validation=validation_df.iloc[:,:-1]
    y_validation=validation_df["Survived"]
    l_predicciones=[]
    #iteramos sobre el numero de filas de nuestra data de prueba
    for i in range(X_test.shape[0]):
        #con el algorimo que creamos anteriormente realizamos la prediccion para cada registro
        prediction = knn(train_df, X_test.iloc[i].tolist(), k)
        #añadimos estra prediccion a nuestra lista de predicciones
        l_predicciones.append(prediction)
    # con ayuda de la funcion accuracy_score obtenemos esta metrica de nuestro conjunto de prueba
    print("accuracy conjunto test:",accuracy_score(y_test,l_predicciones))
    #hacemos lo mismo con classification_report que nos devolvera el resto de las metricas
    report = classification_report(y_test,l_predicciones)
    #imprimimos el reporte de las metricas
    print("Metricas conjunto de prueba\n")
    print(report) 
        

    l_predicciones2=[]
    #hacemos el mismo proceso pero para nuestro conjunto de validacion
    #iteramos sobre el numero de filas de nuestra data de validacion
    for i in range(X_validation.shape[0]):
        #con el algorimo que creamos anteriormente realizamos la prediccion para cada registro
        prediction = knn(train_df, X_validation.iloc[i].tolist(), k)
        #añadimos estra prediccion a nuestra lista de predicciones
        l_predicciones2.append(prediction)
    # con ayuda de la funcion accuracy_score obtenemos esta metrica de nuestro conjunto de validacion
    print("accuracy conjunto de validacion:",accuracy_score(y_validation,l_predicciones2))
    report = classification_report(y_validation,l_predicciones2)
    #hacemos lo mismo con classification_report que nos devolvera el resto de las metricas
    print("Metricas conjunto de validacion\n")
    #imprimimos el reporte de las metricas
    print(report)
    print("\n")
    #utilizamos la funcion confusion matrix para obtener los valores de la matriz de confusion y los almacenamos
    list_cm.append(confusion_matrix(y_validation,l_predicciones2))

# Creamos una figura con n subplots
fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(12, 6))

# Mostramos las matrices de confusión en subplots con ayuda de la lista generada anteriormente

for i in range(n):
    """Generamos y mostrams un conjunto de gráficos de matrices de confusión
    Itera a través de un rango de valores desde 0 hasta n o numero de purebas
    Para cada iteración, genera una matriz de confusión y la visualiza como un mapa de calor
    Anota los valores dentro del mapa de calor
    Utiliza la paleta de colores "Blues" para representar la matriz
    Establece etiquetas en los ejes X e Y para indicar las clases "No sobrevive" y "Sobrevive"
    Asigna títulos a cada gráfico que indican el número de la matriz de confusión"""
    sns.heatmap(list_cm[i], annot=True, cmap="Blues", fmt="d", ax=axes[i],xticklabels=["No sobrevive", "Sobrevive"], yticklabels=["No sobrevive", "Sobrevive"])
    axes[i].set_title('Matriz de Confusión ' + str(i + 1))
#se agrega espacio entre los subplots y se despliega la grafica
plt.tight_layout()
#se muestra el grafico
plt.show()
