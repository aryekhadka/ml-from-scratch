import numpy as np

def LinearRegression():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 5, 8, 11, 14])
    # Adding a bias to x
    ones = np.ones((x.shape[0], 1))
    x = np.hstack((x.reshape(-1, 1), ones))
    # Calculating the weight or 'training' the model
    w = (np.linalg.inv(x.T @ x) @ (x.T @ y))
    # Predicrting new values using y = wx + b no b here since we already added the bias to x
    print(np.array([6, 1] @ w))
    


if __name__ == '__main__':
    LinearRegression()
