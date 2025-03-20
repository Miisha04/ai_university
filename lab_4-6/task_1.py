import numpy as np
import matplotlib.pyplot as plt

# Глобальные общие ограничения
X_MIN = 0    # x >= 0
X_MAX = 1    # x <= 1
Y_MIN = -0.5 # y >= -0.5
Y_MAX = 1.5  # y <= 1.5

N = 100

def generate_C1(N):
    # Зона 3: y < x + 0.5, y > x - 0.5, y > -x + 1.5, x >= 0.5
    N3 = N // 2
    x3 = np.random.uniform(0.5, X_MAX, N3)  # x от 0.5 до 1
    y3_lower = np.maximum(np.maximum(x3 - 0.5, -x3 + 1.5), Y_MIN) 
    y3_upper = np.minimum(x3 + 0.5, Y_MAX)                          
    y3 = np.random.uniform(y3_lower, y3_upper, N3)

    # Зона 5: y < -x + 0.5, y > x - 0.5, y < x + 0.5, x в [0, 0.5]
    N5 = N - N3
    x5 = np.random.uniform(X_MIN, 0.5, N5)
    y5_lower = np.maximum(x5 - 0.5, Y_MIN)
    y5_upper = np.minimum(np.minimum(-x5 + 0.5, x5 + 0.5), Y_MAX)
    y5 = np.random.uniform(y5_lower, y5_upper, N5)

    x = np.hstack([x3, x5])
    y = np.hstack([y3, y5])

    C1 = np.vstack([x, y])
    return C1

def generate_C2(N):
    # Зона 1: y < -x + 1.5, y > -x + 0.5, y < x + 0.5, y > x -0.5
    N1 = N // 2
    x1 = np.random.uniform(X_MIN, X_MAX, N1)
    y1_lower = np.maximum(np.maximum(-x1 + 0.5, x1 - 0.5), Y_MIN)
    y1_upper = np.minimum(np.minimum(-x1 + 1.5, x1 + 0.5), Y_MAX)
    y1 = np.random.uniform(y1_lower, y1_upper, N1)

    # Зона 4: y < x - 0.5, y < -x + 1.5, y > -x + 0.5
    N4 = N - N1
    x4 = np.random.uniform(0.5, X_MAX, N4)
    y4_lower = np.maximum(-x4 + 0.5, Y_MIN)
    y4_upper = np.minimum(np.minimum(-x4 + 1.5, x4 - 0.5), Y_MAX) 
    y4 = np.random.uniform(y4_lower, y4_upper, N4)

    x = np.hstack([x1, x4])
    y = np.hstack([y1, y4])

    C2 = np.vstack([x, y])
    return C2


def act1(x):
    return 0 if x <= 0 else 1

def act2(x):
    return 1 if x == 1 else 0

def go(C):
    w1 = [1, 1, -1.5]  # для y = -x + 1.5
    w2 = [1, 1, -0.5]  # для y = -x + 0.5
    w3 = [-1, 1, -0.5] # для y = x + 0.5
    w4 = [-1, 1, 0.5]  # y = x - 0.5
    
    w_hidden = np.array([w1, w2, w3, w4])
    w_out = np.array([1, -1, 1, 1])

    for i in range(C.shape[1]):
        x = np.array([C[0][i], C[1][i], 1])
        sum = np.dot(w_hidden, x)
        out = [act1(x) for x in sum] 
        out = np.array(out) 
        sum_out = np.dot(w_out, out)  #
        y = act2(sum_out)

        if y == 0:
            plt.text(C[0][i] + 0.02, C[1][i], 'C2', fontsize=8, color='blue') 
        else:
            plt.text(C[0][i] + 0.02, C[1][i], 'C1', fontsize=8, color='red') 



def main():
    C1 = generate_C1(N)
    C2 = generate_C2(N)
    go(C1)
    go(C2)

    f1_x = [0, 1]           # y = -x + 1.5
    f1_y = [1.5, 0.5]
    f2_x = [0, 1]           # y = -x + 0.5
    f2_y = [0.5, -0.5]
    f3_x = [0, 1]           # y = x + 0.5
    f3_y = [0.5, 1.5]
    f4_x = [0, 1]           # y = x - 0.5
    f4_y = [-0.5, 0.5]


    plt.plot(f1_x, f1_y, label='y = -x + 1.5')
    plt.plot(f2_x, f2_y, label='y = -x + 0.5')
    plt.plot(f3_x, f3_y, label='y = x + 0.5')
    plt.plot(f4_x, f4_y, label='y = x - 0.5')

    plt.scatter(C1[0][:], C1[1][:], s=10, c='red', label='C1 (Zones 3, 5)')
    plt.scatter(C2[0][:], C2[1][:], s=10, c='blue', label='C2 (Zones 1, 4)')

    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(X_MIN, X_MAX)
    plt.ylim(Y_MIN, Y_MAX)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
