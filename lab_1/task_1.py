from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


#5variant
# Создайте с использованием NumPy массив из 60 элементов (от 10 до 70), пере-
# форматируйте в формат (6 × 10), выполните слайсинг с шагом 2 по первой оси
# и 3 по второй

start = 10
end = 70
arr = np.array(range(start, end))
arr = arr.reshape((6,10))

result = arr[::2, ::3]

print(result)