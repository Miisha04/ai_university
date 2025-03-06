import numpy as np
import time

# 5variant
# Напишите функцию, которая по заданной длине последовательности случайных
# чисел находит максимальный элемент. Реализуйте задачу с и без помощи
# NumPy, сравните время выполнения

dlina_posl = 100_000_000

def find_max_pd(arr: np.ndarray) -> int | float:
    return arr.max()

def find_max(arr: list) -> int | float:
    return max(arr)

def main():
    arr = np.random.randint(low=0, high=500, size=dlina_posl)

    start_time = time.time()
    print(f'max value pd: {find_max_pd(arr)}')
    end_time = time.time()
    print(f'Время выполнения для find_max_pd: {end_time - start_time} секунд')

    start_time = time.time()
    print(f'max value list: {find_max(arr.tolist())}')
    end_time = time.time()
    print(f'Время выполнения для find_max (с list): {end_time - start_time} секунд')

if __name__ == "__main__":
    main()
