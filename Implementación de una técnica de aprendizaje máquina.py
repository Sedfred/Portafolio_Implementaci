
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
df=pd.read_csv("C:/Users/Admin/Downloads/titanic.csv")
print("Data sin limpieza:", df.shape)
df.drop(["PassengerId", "Name","Ticket","Cabin"], axis=1, inplace=True)
print("Valores vacios en la data:\n",df.isnull().sum())

mage = df['Age'].median()
df['Age'].fillna(mage, inplace=True)
df.dropna(inplace=True)
print("Valores vacios en la data despues de la limpeza:\n",df.isnull().sum())

label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['Embarked'] = label_encoder.fit_transform(df['Embarked'])
scaler = MinMaxScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])
print("normalizacion de la data")

df=df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
       'Embarked','Survived']]
train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)
test_df, validation_df = train_test_split(temp_df, test_size=0.5, random_state=42)
print("Tamaño del conjunto de entrenamiento:", len(train_df))
print("Tamaño del conjunto de prueba:", len(test_df))
print("Tamaño del conjunto de validación:", len(validation_df))

def distancia_euc(p1, p2):
  sum_disc = 0.0
  for cord1, cord2 in zip(p1, p2):
    sum_disc += (cord1 - cord2) ** 2
  distancia_euc = math.sqrt(sum_disc)
  return distancia_euc

def knn(data, point, k):
  dist = []
  for _, i in data.iterrows():
      distancia = distancia_euc(point, i[:-1])
      label = i[-1]
      dist.append((distancia, label))
  dist.sort()
  label_kn = []
  for _, l in dist[:k]:
      label_kn.append(l)
  list_lable = {}
  for label in label_kn:
      if label in list_lable:
          list_lable[label] += 1
      else:
          list_lable[label] = 1
  
  most_common_label = max(list_lable, key=list_lable.get)
  return most_common_label

training_data = train_df
k = 6

X_test=test_df.iloc[:,:-1]
y_test=test_df["Survived"]
X_validation=validation_df.iloc[:,:-1]
y_validation=validation_df["Survived"]
l_predicciones=[]
for i in range(X_test.shape[0]):
  prediction = knn(training_data, X_test.iloc[i].tolist(), k)
  l_predicciones.append(prediction)

from sklearn.metrics import accuracy_score
print("++++++++PRIMERA PUREBA+++++++++")
print("accuracy conjunto test:",accuracy_score(y_test,l_predicciones))

l_predicciones2=[]
for i in range(X_validation.shape[0]):
  prediction = knn(training_data, X_validation.iloc[i].tolist(), k)
  l_predicciones2.append(prediction)

from sklearn.metrics import accuracy_score
print("accuracy conjunto de validacion:",accuracy_score(y_validation,l_predicciones2))

print("++++++++SEGUNDA PUREBA+++++++++")
train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)
test_df, validation_df = train_test_split(temp_df, test_size=0.5, random_state=88)
print("Tamaño del conjunto de entrenamiento:", len(train_df))
print("Tamaño del conjunto de prueba:", len(test_df))
print("Tamaño del conjunto de validación:", len(validation_df))

X_test=test_df.iloc[:,:-1]

y_test=test_df["Survived"]

X_validation=validation_df.iloc[:,:-1]


y_validation=validation_df["Survived"]


l_predicciones=[]
for i in range(X_test.shape[0]):
  prediction = knn(training_data, X_test.iloc[i].tolist(), k)
  l_predicciones.append(prediction)

from sklearn.metrics import accuracy_score

print("accuracy conjunto test:",accuracy_score(y_test,l_predicciones))

l_predicciones2=[]
for i in range(X_validation.shape[0]):
  prediction = knn(training_data, X_validation.iloc[i].tolist(), k)
  l_predicciones2.append(prediction)

from sklearn.metrics import accuracy_score
print("accuracy conjunto de validacion:",accuracy_score(y_validation,l_predicciones2))