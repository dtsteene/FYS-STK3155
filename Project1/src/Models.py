import numpy as np
from sklearn.linear_model import Lasso as LassoSKL


class Model:
    "Baseclass for regression models."

    def __init__(self) -> None:
        self.fitted = False

    def predict(self, X: np.array) -> np.array:
        """Predict values from regression.

        inputs:
            X (np.array): Design matrix.
        returns:
            (np.array) Predicted values.
        """
        if not self.fitted:
            raise ValueError("Model can not predict before being fitted")

        return X @ self.beta

    @staticmethod
    def create_X(x: np.array, y: np.array, n: int) -> np.array:
        """Create design matrix for polynomial regression.

        inputs:
            x (np.array): Values in x direction
            y (np.array): Values in y direction
            n (int): Maximum polynomial degree
        returns:
            (np.array) Design matrix
        """
        if len(x.shape) > 1:
            x = np.ravel(x)
            y = np.ravel(y)

        N = len(x)
        lgth = (n + 1) * (n + 2) // 2
        X = np.ones((N, lgth))
        for i in range(1, n + 1):
            q = i * (i + 1) // 2
            for k in range(i + 1):
                X[:, q + k] = (x ** (i - k)) * (y**k)

        return X


class Ridge(Model):
    "Model for Ridge regression."

    def __init__(self, lmbd: float = None) -> None:
        """Initialize the model for Ridge regression.

        inputs:
            lmbd (float): Regularization factor
        """
        super().__init__()
        self.modelName = "Ridge"
        self.lmbd = lmbd

    def fit(self, X: np.array, y: np.array, lmbd: float = None) -> np.array:
        """Fit the model from the data

        inputs:
            X (np.array): Design matrix
            y (np.array): Response variable
            lmbd (float): Regularization factor
        returns:
            (np.array) Fitted regression parameter
        """
        if lmbd is not None:
            self.lmbd = lmbd
        if self.lmbd is None:
            raise ValueError("No lambda provided")

        Identity = np.identity(X.shape[1])
        self.beta = np.linalg.pinv(X.T @ X + self.lmbd * Identity) @ X.T @ y
        self.fitted = True

        return self.beta


class OLS(Model):
    "Model for Ordinary Least Squares regression"

    def __init__(self) -> None:
        "Initialize the model for OLS regression."
        super().__init__()
        self.name = "OLS"

    def fit(self, X: np.array, y: np.array) -> np.array:
        """Fit the model from the data

        inputs:
            X (np.array): Design matrix
            y (np.array): Response variable
        returns:
            (np.array) Fitted regression parameter
        """
        self.beta = np.linalg.pinv(X.T @ X) @ X.T @ y
        self.fitted = True

        return self.beta


class Lasso(Model):
    "Model for Lasso regression"

    def __init__(self, lmbd=None) -> None:
        """Initialize the model for Ridge regression.

        inputs:
            lmbd (float): Regularization factor
        """
        super().__init__()
        self.modelName = "Lasso"
        self.lmbd = lmbd
        self.lasso_model: LassoSKL = None

    def fit(self, X: np.array, y: np.array, lmbd: float = None) -> np.array:
        """Fit the model from the data

        inputs:
            X (np.array): Design matrix
            y (np.array): Response variable
            lmbd (float): Regularization factor
        returns:
            (np.array) Fitted regression parameter
        """
        if lmbd is not None:
            self.lmbd = lmbd
        if self.lmbd is None:
            raise ValueError("No lambda/alpha provided")

        self.lasso_model = LassoSKL(alpha=lmbd, max_iter=100000)
        self.lasso_model.fit(X, y)
        self.beta = np.hstack((self.lasso_model.intercept_, self.lasso_model.coef_))
        self.fitted = True

        return self.beta

    def predict(self, X: np.array) -> np.array:
        """Predict values from regression.

        inputs:
            X (np.array): Design matrix.
        returns:
            (np.array) Predicted values.
        """
        if not self.fitted:
            raise ValueError("Model can not predict before being fitted")

        return self.lasso_model.predict(X).reshape(-1, 1)
