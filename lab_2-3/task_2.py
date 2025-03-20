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

    X_train = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]])
    y_train = np.array([linear_function(x1,x2,x3) for x1, x2, x3 in X_train])

    epochs = 1000
    learning_rate = 0.01


    for epoch in range(epochs):
        y_pred = forward_pass(X_train)
        loss = compute_loss(y_train, y_pred)

        update_weights(X_train, y_train, y_pred, learning_rate)

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss:.8f}, weights {weights}, bias{bias}')


    X_test = np.array([[7,8,9],[8,9,10]])

    predictions = forward_pass(X_test)

    y_true = np.array([linear_function(x1,x2,x3) for x1,x2,x3 in X_test])
    
    print('Predictions:', predictions,'y_true:', y_true)



if __name__ == "__main__":
    main()