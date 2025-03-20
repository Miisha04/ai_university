import numpy as np

# Функция линейного преобразования
def linear_function(x1, x2, x3) -> int | float:
    return x1 + 2*x2 + 3*x3

# Прямой проход (предсказание)
def forward_pass(X):
    return np.dot(X, weights) + bias

# Функция ошибки (MSE)
def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Обновление весов градиентным спуском
def update_weights(X, y_true, y_pred, learning_rate):
    global weights, bias
    error = y_pred - y_true

    weights_gradient = np.dot(X.T, error) / len(X)
    bias_gradient = np.mean(error)

    weights -= learning_rate * weights_gradient
    bias -= learning_rate * bias_gradient

# Основная функция
def main():
    global weights, bias

    num_experiments = 5  
    learning_rate = 0.01  
    biases_pred = []
    biases_true = []

    for exp in range(1, num_experiments + 1):
        print(f"\n🚀 Эксперимент {exp}")

        # Инициализация случайных весов и биаса
        weights = np.random.rand(3)
        bias = np.random.rand()

        # Генерация случайных данных + фиксированный массив [5, 5, 5]
        X_random = np.random.randint(1, 10, size=(50, 3))
        X_fixed = np.array([[5, 5, 5]])  # Один фиксированный массив
        X_train = np.vstack((X_random, X_fixed))  # Объединение массивов

        y_train = np.array([linear_function(x1, x2, x3) for x1, x2, x3 in X_train])

        # Генерация случайного количества эпох
        epochs = np.random.randint(5000, 8000)
        print(f"🔄 Количество эпох: {epochs}")

        # Обучение модели
        for epoch in range(epochs):
            y_pred = forward_pass(X_train)
            loss = compute_loss(y_train, y_pred)

            update_weights(X_train, y_train, y_pred, learning_rate)

            # Вывод каждые 100 эпох
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss:.8f}, Weights: {weights}, Bias: {bias:.4f}')

        # Генерация случайных данных для тестирования
        X_test = np.random.randint(1, 10, size=(2, 3))
        predictions = forward_pass(X_test)
        y_true = np.array([linear_function(x1, x2, x3) for x1, x2, x3 in X_test])

        print('🔹 Тестовые данные:', X_test)
        print('🔹 Предсказания:', predictions)
        print('🔹 Истинные значения:', y_true)

        # Сохранение предсказанного и реального биаса
        biases_pred.append(bias)
        biases_true.append(0)  # Если в линейной функции нет биаса, то истинное значение 0

    # Вычисление средних ошибок
    biases_pred = np.array(biases_pred)
    biases_true = np.array(biases_true)

    mse = np.mean((biases_pred - biases_true) ** 2)
    mae = np.mean(np.abs(biases_pred - biases_true))
    mean_bias = np.mean(biases_pred)

    # Вывод более понятных метрик
    print(f"\n📊 Среднеквадратичное отклонение биаса (MSE): {mse:.6f}")
    print(f"📊 Средняя абсолютная ошибка (MAE): {mae:.6f}")
    print(f"📊 Среднее предсказанное значение биаса: {mean_bias:.6f}")
    print(f"📊 Средняя ошибка в процентах: {mae * 100:.2f}%")

if __name__ == "__main__":
    main()
