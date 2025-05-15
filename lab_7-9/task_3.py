import numpy as np
from tensorflow.keras.models import Sequential # Обновленный импорт
from tensorflow.keras.layers import Dense      # Обновленный импорт
# from tensorflow.keras.utils import to_categorical # Не используется для регрессии
from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plt
# import matplotlib.colors as mclr # Не используется

# 1. Загрузка набора данных
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# Контрольный вывод размера обучающей и тестовой выборок
print(f"Размер обучающей выборки: {train_data.shape}")
print(f"Размер тестовой выборки: {test_data.shape}")
# Контрольный вывод первых 5 элементов результатирующего вектора тестовой выборки
print(f"Первые 5 целевых значений тестовой выборки: {test_targets[:5]}")

# 2. Нормализация данных
# Вычисление среднего значения по обучающей выборке
mean = train_data.mean(axis=0)
# Вычитание среднего значения из данных обучающей выборки
train_data -= mean
# Определение стандартного отклонения по обучающей выборке
std = train_data.std(axis=0)
# Нормирование данных обучающей выборки
train_data /= std

# Нормирование данных тестовой выборки с использованием среднего и std обучающей выборки
test_data -= mean
test_data /= std

# 3. Определение функции построения и компиляции модели
def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],))) # Указание input_shape для первого слоя
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1)) # Один выходной нейрон для регрессии (без функции активации)
    # Компиляция модели
    # mse: Mean Squared Error (среднеквадратичная ошибка)
    # mae: Mean Absolute Error (средняя абсолютная ошибка)
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

# 4. K-кратная перекрестная проверка
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100 # Можно увеличить для лучшего выявления переобучения, например, до 500
all_mae_histories = []
all_mse_histories = [] # Для хранения истории MSE (loss)

print("\n--- Начало K-кратной перекрестной проверки ---")
for i in range(k):
    print(f'Обработка блока #{i}')
    # Формирование валидационной выборки
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # Формирование обучающей выборки для текущего блока
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    # Построение модели НС
    model = build_model()
    # Обучение модели (verbose=0 отключает вывод логов обучения)
    # Сохраняем историю обучения
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets), # Добавляем валидационные данные для отслеживания метрик на них
                        epochs=num_epochs, batch_size=16, verbose=0) # Увеличен batch_size для ускорения

    # Сохранение истории MAE и MSE для текущего блока
    mae_history = history.history['val_mae']
    mse_history = history.history['val_loss'] # 'loss' на валидации это val_loss
    all_mae_histories.append(mae_history)
    all_mse_histories.append(mse_history)

print("--- K-кратная перекрестная проверка завершена ---")

# Расчет средних значений MAE и MSE по всем блокам для каждой эпохи
average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
average_mse_history = [
    np.mean([x[i] for x in all_mse_histories]) for i in range(num_epochs)]


# 7. Построение графиков средней квадратичной ошибки (MSE) и средней абсолютной ошибки (MAE)
#    во время обучения для моделей каждого блока, а также усредненные графики.

plt.figure(figsize=(15, 10))

# Графики MAE для каждого блока
plt.subplot(2, 2, 1)
for i, mae_hist in enumerate(all_mae_histories):
    plt.plot(range(1, len(mae_hist) + 1), mae_hist, label=f'MAE блока {i+1} (вал.)')
plt.title('MAE на валидации для каждого блока')
plt.xlabel('Эпохи')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)

# Графики MSE (loss) для каждого блока
plt.subplot(2, 2, 2)
for i, mse_hist in enumerate(all_mse_histories):
    plt.plot(range(1, len(mse_hist) + 1), mse_hist, label=f'MSE блока {i+1} (вал.)')
plt.title('MSE на валидации для каждого блока')
plt.xlabel('Эпохи')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)

# Усредненный график MAE
plt.subplot(2, 2, 3)
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history, 'b-', label='Усредненная MAE (вал.)')
# Также добавим MAE на обучающей выборке из последнего history для сравнения (можно усреднить и их, но для простоты)
if 'mae' in history.history:
    plt.plot(range(1, len(history.history['mae']) + 1), history.history['mae'], 'r--', label='MAE на обучении (последний блок)')
plt.title('Усредненная MAE на валидации')
plt.xlabel('Эпохи')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)

# Усредненный график MSE
plt.subplot(2, 2, 4)
plt.plot(range(1, len(average_mse_history) + 1), average_mse_history, 'b-', label='Усредненная MSE (вал.)')
# Также добавим MSE на обучающей выборке из последнего history
if 'loss' in history.history:
    plt.plot(range(1, len(history.history['loss']) + 1), history.history['loss'], 'r--', label='MSE на обучении (последний блок)')
plt.title('Усредненная MSE на валидации')
plt.xlabel('Эпохи')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 8. Выявление точки переобучения
# Точка переобучения обычно наблюдается, когда ошибка на валидационной выборке (val_loss или val_mae)
# начинает расти, в то время как ошибка на обучающей выборке (loss или mae) продолжает уменьшаться.
# Посмотрите на усредненные графики (особенно на "Усредненная MAE на валидации" и "Усредненная MSE на валидации").
# Эпоха, после которой синяя линия (валидация) начинает стабильно подниматься,
# может считаться точкой начала переобучения.
# Например, если val_mae начинает расти после 50-й эпохи, то оптимальное количество эпох может быть около 50.
# Для более точного определения, возможно, потребуется больше эпох (num_epochs).
print("\n--- 8. Выявление точки переобучения ---")
print("Проанализируйте графики MAE и MSE на валидационной выборке.")
print("Точка переобучения - это эпоха, после которой ошибка на валидации перестает уменьшаться или начинает расти.")
# Автоматическое определение точки переобучения (примерный подход)
if len(average_mae_history) > 10: # Убедимся, что достаточно данных
    # Находим эпоху, где MAE на валидации минимальна
    optimal_epochs_mae = np.argmin(average_mae_history) + 1
    print(f"Примерная оптимальная точка по MAE (минимум на валидации): {optimal_epochs_mae} эпох.")
    # Можно также искать, где производная меняет знак, но минимум проще.
else:
    optimal_epochs_mae = num_epochs # Если данных мало, используем все эпохи
    print("Недостаточно данных для точного определения точки переобучения, используем num_epochs.")


# 9. Обучение построенной модели нейронной сети сразу на всех данных обучающей выборки
print("\n--- 9. Обучение модели на всех обучающих данных ---")
model_final = build_model()
# Обучаем на всех train_data, выделяя 20% для валидации внутри fit
# Используем 'optimal_epochs_mae' или фиксированное количество, если не уверены
# Для примера, используем немного больше эпох, чем предполагаемая точка переобучения,
# или фиксированное значение, если оно было хорошо подобрано.
# Здесь для примера возьмем optimal_epochs_mae, если оно разумно, или num_epochs/2
# Важно: не используйте слишком много эпох, чтобы избежать сильного переобучения.
epochs_for_final_model = optimal_epochs_mae if optimal_epochs_mae < num_epochs else num_epochs // 2
if epochs_for_final_model < 10: # Минимальное количество эпох
    epochs_for_final_model = max(10, num_epochs // 2)

print(f"Обучение финальной модели на {epochs_for_final_model} эпохах...")
history_final = model_final.fit(train_data, train_targets,
                                epochs=epochs_for_final_model, # Используем определенное количество эпох
                                batch_size=16,
                                validation_split=0.2, # 20% данных для валидации
                                verbose=1) # Включим вывод для финального обучения

# Графики обучения для финальной модели
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_final.history['loss'], label='MSE на обучении (финал)')
plt.plot(history_final.history['val_loss'], label='MSE на валидации (финал)')
plt.title('MSE для финальной модели')
plt.xlabel('Эпохи')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history_final.history['mae'], label='MAE на обучении (финал)')
plt.plot(history_final.history['val_mae'], label='MAE на валидации (финал)')
plt.title('MAE для финальной модели')
plt.xlabel('Эпохи')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# 10. Тестирование обученной НС на тестовой выборке
print("\n--- 10. Тестирование обученной НС на тестовой выборке ---")
test_mse_score, test_mae_score = model_final.evaluate(test_data, test_targets, verbose=0)
print(f"Тестовая MSE (среднеквадратичная ошибка): {test_mse_score:.2f}")
print(f"Тестовая MAE (средняя абсолютная ошибка): {test_mae_score:.2f}")
print(f"Это означает, что в среднем предсказания модели отклоняются от фактических цен на ${test_mae_score*1000:.0f}.")


# 11. Вывод полученных результатов предсказаний и желаемых откликов для тестовой выборки
print("\n--- 11. Сравнение предсказаний и желаемых откликов (первые 10) ---")
predictions = model_final.predict(test_data).flatten() # .flatten() для преобразования в 1D массив

print("Предсказание | Факт. значение | Разница")
print("-----------------------------------------")
for i in range(min(10, len(predictions))): # Выводим первые 10 или меньше, если их меньше
    print(f"{predictions[i]:11.2f} | {test_targets[i]:14.2f} | {predictions[i] - test_targets[i]:8.2f}")

# Выводы о точности предсказания:
# MAE на тестовой выборке дает представление о среднем абсолютном отклонении предсказаний от реальных значений.
# Например, MAE = 2.5 означает, что в среднем модель ошибается на $2500 (т.к. цены в тысячах долларов).
# Чем меньше MAE и MSE, тем лучше модель. Сравните с MAE, полученной на валидации во время K-fold.
# Если тестовая MAE значительно выше валидационной, это может указывать на переобучение или на то,
# что тестовая выборка сильно отличается от обучающей/валидационной.

# 12. Отображение на одном графике полученных результатов предсказаний и желаемых откликов
print("\n--- 12. График предсказаний vs фактических значений ---")
plt.figure(figsize=(10, 6))
plt.scatter(test_targets, predictions, alpha=0.5, label='Предсказания vs Факт')
# Добавим линию y=x для идеальных предсказаний
min_val = min(np.min(test_targets), np.min(predictions))
max_val = max(np.max(test_targets), np.max(predictions))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Идеальное предсказание')
plt.xlabel("Фактические значения (тыс. $)")
plt.ylabel("Предсказанные значения (тыс. $)")
plt.title("Сравнение фактических и предсказанных значений на тестовой выборке")
plt.legend()
plt.grid(True)
plt.show()

print("\n--- Анализ завершен ---")

