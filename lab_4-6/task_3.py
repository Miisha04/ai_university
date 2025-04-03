import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris

# Загрузка данных Iris из sklearn
iris = load_iris()
dataframe = pd.DataFrame(data=iris.data, columns=iris.feature_names)
dataframe['target'] = iris.target_names[iris.target]
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
print(X.shape)
Y = dataset[:,4]

#преобразование выходных атрибутов из вектора в матрицу
#переход от текстовых меток к категориальному вектору
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = to_categorical(encoded_Y)

#определение базовой архитектуры сети
model = Sequential()
#основным строительным блоком нейронных сетей является слой
#методика ГО заключается в объединении простых слоев, реализующих некоторую форму поэтапной очистки данных
#сеть состоит из 2 полносвязных слоев Dense
#функция активации softmax используется для распределения результата по вероятностям
#выходной слой — 3-переменный слой softmax layer, возвращающий массив с 3 оценками вероятностей (в сумме дающих 1)
#каждая оценка определяет вероятность принадлежности текущего изображения к одному из 3 классов цветов
model.add(Dense(4, activation="relu"))
model.add(Dense(3, activation="softmax"))

#настройка параметров сети: функции потерь, оптимизатора, метрики для мониторинга на этапах обучения и тестирования
#функции потерь categorical_crossentropy определяет расстояние между распределениями вероятностей:
#между распределением вероятности на выходе сети и истинным распределением меток
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

#обучение сети
model.fit(X, dummy_y, epochs=200, batch_size=10, validation_split=0.1)
#в процессе обучения отображаются четыре величины: потери и точность сети на обучающей и валидационной выборках

#преобразование тестовых данных
x = np.expand_dims([6,7,2,3], axis=0)
#получение предсказания модели
res = model.predict(x)
print(res)

def form_answer(res):
    probabilities = res[0]
    index_max = np.argmax(probabilities) 
    class_names = ["Iris Setosa", "Iris Versicolor", "Iris Virginica"]

    prob_text = ", ".join(
        [f"с вероятностью {prob:.0%} – к классу {name}" for prob, name in zip(probabilities, class_names)]
    )

    return (
        f"{prob_text}. Наиболее вероятно, что представленный экземпляр относится к классу {class_names[index_max]}."
    )

# Вывод результата
print(form_answer(res))
