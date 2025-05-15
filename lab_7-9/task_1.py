import keras
import pandas
import numpy as np
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.metrics import Recall


# 5 вариант:
# количество фичей: 15–25 → выберем 20
# количество скрытых слоев: 2–4 → добавим 3
# оптимизатор: Adagrad
# метрика: recall

# Загружаем датасет
dataframe = pandas.read_csv("sonar.csv", header=None)
dataset = dataframe.values
X = dataset[:, 0:20].astype(float)  # <-- берем 20 фичей вместо 60
print(X)
Y = dataset[:, 60]

# Кодируем метки
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# Создание модели
model = Sequential()
# 1 скрытый слой
model.add(Dense(128, input_dim=20, kernel_initializer="he_normal", activation="relu"))
# 2 скрытый слой
model.add(Dense(64, kernel_initializer="he_normal", activation="relu"))
# 3 скрытый слой
model.add(Dense(32, kernel_initializer="he_normal", activation="relu"))
# Выходной слой
model.add(Dense(1, kernel_initializer='he_normal', activation="sigmoid"))

# Компиляция модели с нужным оптимизатором и метрикой
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['Recall'])

# Обучение модели
model.fit(X, encoded_Y, epochs=150, batch_size=10, validation_split=0.1)

# Визуализация архитектуры модели
keras.utils.plot_model(model, "my_first_model.png")
keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)


import matplotlib.pyplot as plt

# Выбираем две фичи для отображения
x_feature = 0
y_feature = 1

# Получаем координаты точек
x_vals = X[:, x_feature]
y_vals = X[:, y_feature]

# Разделяем точки по классам
class_0 = encoded_Y == 0
class_1 = encoded_Y == 1

# Визуализация
plt.figure(figsize=(8, 6))
plt.scatter(x_vals[class_0], y_vals[class_0], color='blue', label='Класс 0', alpha=0.7)
plt.scatter(x_vals[class_1], y_vals[class_1], color='red', label='Класс 1', alpha=0.7)

plt.title("Графическое отображение классов по двум признакам")
plt.xlabel(f'Признак {x_feature}')
plt.ylabel(f'Признак {y_feature}')
plt.legend()
plt.grid(True)
plt.show()



# рекол не особо влияет, адам лучше адаграда, кол-во эпох не меньше 150
#сделать 3 эксперимента

