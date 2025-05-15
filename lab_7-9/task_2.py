import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from matplotlib.lines import Line2D

# Загрузка и кодировка данных
data = pd.read_csv("bank.csv", sep=';')

label_encoders = {}
for column in data.columns:
    if data[column].dtype == 'object':
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

# Фичи и таргет
X = data.drop("y", axis=1)
y = data["y"]

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/6, random_state=42)

# Нормализация
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Модель нейронной сети
model = Sequential([
    Dense(32, input_dim=X_train.shape[1], activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Обучение модели
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=900, verbose=1)

# Оценка модели
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nТочность на тестовой выборке: {accuracy:.4f}")

# Архитектура модели
plot_model(model, to_file='model.png', show_shapes=True)

# Графики потерь и точности
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Обучающая')
plt.plot(history.history['val_loss'], label='Валидационная')
plt.title('График потерь')
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Обучающая')
plt.plot(history.history['val_accuracy'], label='Валидационная')
plt.title('График точности')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend()

plt.tight_layout()
plt.show()

feature1 = 0  # Например: 'age'
feature2 = 5  # Например: 'balance'
feature1_name = X.columns[feature1]
feature2_name = X.columns[feature2]

# Предсказания модели
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Добавим немного шума, чтобы избежать полного наложения (jitter)
jitter_strength = 0.1
x1 = X_test[:, feature1] + np.random.normal(0, jitter_strength, size=len(X_test))
x2 = X_test[:, feature2] + np.random.normal(0, jitter_strength, size=len(X_test))

# Настройка параметров отображения
colors = ['green' if y == 1 else 'red' for y in y_test]
markers = ['o' if y_pred[i] == y_test.values[i] else 'x' for i in range(len(y_test))]

# Размеры точек по классам
sizes = [100 if label == 1 else 40 for label in y_test]

# Построение графика
plt.figure(figsize=(10, 7))
for i in range(len(x1)):
    plt.scatter(
        x1[i], x2[i],
        c=colors[i],
        marker=markers[i],
        edgecolors='black',
        s=sizes[i],
        alpha=0.6,
        linewidth=0.6
    )

# Легенда
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Класс 1 (Yes)', markerfacecolor='green', markersize=10, markeredgecolor='k'),
    Line2D([0], [0], marker='o', color='w', label='Класс 0 (No)', markerfacecolor='red', markersize=6, markeredgecolor='k'),
    Line2D([0], [0], marker='x', color='red', label='Неправильное предсказание', markersize=10)
]
plt.legend(handles=legend_elements, loc='best')

plt.xlabel(feature1_name)
plt.ylabel(feature2_name)
plt.title('Тестовая выборка: визуализация классов (разный размер точек)')
plt.grid(True)
plt.tight_layout()
plt.show()