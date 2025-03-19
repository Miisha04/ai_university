import numpy as np
import random

#5variant
# Выполнить следующее преобразование трех произвольных действительных чи-
# сел: x1 + 2x2 + 3x3

def estim(x1,x2,x3):
    return x1 + 2*x2 + 3*x3


def relu(x):
    return np.maximum(0, x)

def go(x1,x2,x3) -> int | float:
    arr = np.array([x1,x2,x3])

    #h1 = x1 + 2x2 
    #h2 = 3x3
    #res = h1+h2

    weights_hidden = np.array([[1,2,0],[0,0,3]])
    hidden_layer = np.dot(weights_hidden, arr)
    hidden_layer = relu(hidden_layer)
    weights_output = np.array([1,1])
    output = np.dot(weights_output, hidden_layer)

    return output


def main():

    for _ in range(5):
        x1 = random.randint(0, 10)
        x2 = random.randint(0, 10)
        x3 = random.randint(0, 10)
        res = go(x1,x2,x3)

        expected = estim(x1,x2,x3)

        if res == expected:
            print(f"Good answer -> {res} == {expected}, for x1={x1}, x2={x2}, x3={x3}")
        else:
            print(f"Bad answer -> {res} != {expected}")

if __name__ == "__main__":
    main()
