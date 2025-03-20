import numpy as np


#5variant
# Выполнить следующее преобразование трех произвольных действительных чи-
# сел: x1 + 2x2 + 3x3

weights = np.random.rand(3)
bias = 0


def linear_function(x1,x2,x3) -> int | float:
    return x1 + 2*x2 + 3*x3


def forward_pass(X):
    return np.dot(X, weights) + bias


def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def update_weights(X, y_true, y_pred, learning_rate):
    global weights, bias

    error = y_pred - y_true

    weights_gradient = np.dot(X.T, error) / len(X)

    bias_gradient = np.mean(error)

    weights -= learning_rate * weights_gradient
    bias -= learning_rate * bias_gradient


def main():
    global weights, bias

    num_experiments = 5  
    epochs = np.random.randint(800, 1501)  
    learning_rate = 0.01 

    for exp in range(1, num_experiments + 1):
        print(f"\n Эксперимент {exp}")


        weights = np.random.rand(3)
        bias = np.random.rand()

        X_train = np.random.randint(1, 10, size=(5, 3))
        y_train = np.array([linear_function(x1, x2, x3) for x1, x2, x3 in X_train])

        for epoch in range(epochs):
            y_pred = forward_pass(X_train)
            loss = compute_loss(y_train, y_pred)

            update_weights(X_train, y_train, y_pred, learning_rate)

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss:.8f}')

        X_test = np.random.randint(1, 10, size=(2, 3))
        predictions = forward_pass(X_test)
        y_true = np.array([linear_function(x1, x2, x3) for x1, x2, x3 in X_test])

        print('Предсказания:', predictions)
        print('Истинные значения:', y_true)



if __name__ == "__main__":
    main()