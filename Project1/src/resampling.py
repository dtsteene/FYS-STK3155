# here we can write our funcs such that we take a sklear model as input
from Models import OLS, Ridge, Lasso, Model
from Data import Data
import numpy as np
from metrics import MSE, mean_MSE, get_bias, get_variance
from sklearn.utils import resample

from sklearn.model_selection import cross_val_score, KFold as skKfold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.linear_model import Lasso as LassoSKL
from sklearn.linear_model import Ridge as RidgeSKL
from sklearn.linear_model import LinearRegression as OLSSKL
from Kfold import KFold


def bootstrap_degrees(
    data: Data, n_bootstraps: int, model: Model = OLS(), maxDim: int = None
) -> tuple[np.array, np.array, np.array]:
    """Perform bootstrap resampling.

    inputs:
        data (Data): Data for regression analysis
        n_bootstraps (int): Number of bootstraps
        model (Model): Regression model to apply
        maxDim (int): Maximal polynomial degree
    returns:
        (tuple[np.array, np.array, np.array]) of (Error, Bias, Variance)
    """
    polyDegrees = range(1, maxDim + 1)
    n_degrees = len(polyDegrees)

    error = np.zeros(n_degrees)
    bias = np.zeros(n_degrees)
    variance = np.zeros(n_degrees)
    pbar = tqdm(
        total=len(polyDegrees) * n_bootstraps,
        desc=f"Bootstrap {model.__class__.__name__}",
    )
    scaler = StandardScaler()
    for j, dim in enumerate(polyDegrees):
        X_train = model.create_X(data.x_train, data.y_train, dim)
        X_test = model.create_X(data.x_test, data.y_test, dim)

        scaler = scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        z_test, z_train = data.z_test, data.z_train
        z_pred = np.empty((z_test.shape[0], n_bootstraps))

        for i in range(n_bootstraps):
            X_, z_ = resample(X_train, z_train)
            beta = model.fit(X_, z_)
            # z_pred[:, i] = model.predict(X_test).ravel()
            z_pred[:, i] = (X_test @ beta).ravel()
            pbar.update(1)

        error[j] = mean_MSE(z_test, z_pred)
        bias[j] = get_bias(z_test, z_pred)
        variance[j] = get_variance(z_pred)

    return error, bias, variance


def bootstrap_lambdas(
    data: Data,
    n_bootstraps: int,
    model: Ridge | Lasso = Ridge(),
    lmbds: np.array = np.logspace(-3, 5, 10),
    maxDim: int = 15,
    **kwargs,
) -> tuple[np.array, np.array, np.array]:
    """Perform bootstrap resampling.

    inputs:
        data (Data): Data for regression analysis
        n_bootstraps (int): Number of bootstraps
        model (Model): Regression model to apply
        maxDim (int): Maximal polynomial degree
    returns:
        (tuple[np.array, np.array, np.array]) of (Error, Bias, Variance)
    """
    polyDegrees = range(1, maxDim + 1)
    n_degrees = len(polyDegrees)
    n_lmbds = lmbds.size

    error = np.zeros((n_degrees, n_lmbds))
    bias = np.zeros((n_degrees, n_lmbds))
    variance = np.zeros((n_degrees, n_lmbds))

    # for i, dim in tqdm(enumerate(polyDegrees)):
    pbar = tqdm(
        total=n_degrees * n_lmbds * n_bootstraps,
        desc=f"Bootstrap for {model.__class__.__name__}",
    )
    scaler = StandardScaler()

    for i in range(maxDim):
        dim = i + 1
        X_train = model.create_X(data.x_train, data.y_train, dim)
        X_test = model.create_X(data.x_test, data.y_test, dim)

        scaler = scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        z_test, z_train = data.z_test, data.z_train

        z_test = z_test.reshape(z_test.shape[0], 1)
        for j, lambd in enumerate(lmbds):
            z_pred = np.empty((z_test.shape[0], n_bootstraps))
            for k in range(n_bootstraps):
                X_, z_ = resample(X_train, z_train)
                model.fit(X_, z_, lambd)
                z_pred[:, k] = model.predict(X_test).ravel()
                pbar.update(1)

            error[i, j] = mean_MSE(z_test, z_pred)
            bias[i, j] = get_bias(z_test, z_pred)
            variance[i, j] = get_variance(z_pred)

    return error, bias, variance


def sklearn_cross_val(
    data: Data, nfolds: int, model: OLSSKL = OLSSKL(), maxDim: int = 15
) -> tuple[np.array, np.array]:
    """Perform cross validation with Scikit-learn.

    inputs:
        data (Data): Data for regression analysis
        nfolds (int): Number of bootstraps
        model (Model): Regression model to apply
        maxDim (int): Maximal polynomial degree
    returns:
        (tuple[np.array, np.array, np.array]) of (Error, Variance)
    """
    polyDegrees = range(1, maxDim + 1)
    n_degrees = len(polyDegrees)

    error = np.zeros(n_degrees)
    variance = np.zeros(n_degrees)
    dummy_model = Model()
    Kfold = skKfold(nfolds, shuffle=True)
    for i, degree in tqdm(enumerate(polyDegrees), total=maxDim):
        X = dummy_model.create_X(data.x, data.y, degree)
        X = StandardScaler().fit_transform(X)

        scores = cross_val_score(
            model, X, data.z_, scoring="neg_mean_squared_error", cv=Kfold, n_jobs=-1
        )
        error[i] = -scores.mean()
        variance[i] = scores.std()

    return error, variance


def kfold_score_degrees(
    data: Data, kfolds: int, model: OLS = OLS(), maxDim: int = 15
) -> tuple[np.array, np.array]:
    """Perform cross validation.

    inputs:
        data (Data): Data for regression analysis
        kfolds (int): Number of bootstraps
        model (Model): Regression model to apply
        maxDim (int): Maximal polynomial degree
    returns:
        (tuple[np.array, np.array]) of (Error, Variance)
    """
    polyDegrees = range(1, maxDim + 1)
    n_degrees = len(polyDegrees)

    error = np.zeros(n_degrees)
    variance = np.zeros(n_degrees)

    Kfold = KFold(n_splits=kfolds, shuffle=True)

    pbar = tqdm(total=maxDim * kfolds, desc=f"K-Fold {model.__class__.__name__}")
    for i, degree in enumerate(polyDegrees):
        scores = np.zeros(kfolds)

        X = model.create_X(data.x_, data.y_, degree)
        X = StandardScaler().fit_transform(X)

        for j, (train_i, test_i) in enumerate(Kfold.split(X)):
            X_train = X[train_i]
            X_test = X[test_i]
            z_train = data.z_[train_i]
            z_test = data.z_[test_i]

            model.fit(X_train, z_train)

            z_pred = model.predict(X_test)

            scores[j] = MSE(z_pred, z_test)
            pbar.update(1)

        # print(scores)
        error[i] = scores.mean()
        variance[i] = scores.std()
    return error, variance


def sklearn_cross_val_lambdas(
    data: Data,
    kfolds: int,
    model: RidgeSKL | LassoSKL = RidgeSKL(),
    maxDim: int = 15,
    lmbds: np.array = np.logspace(-3, 5, 10),
) -> tuple[np.array, np.array]:
    """Perform cross validation with Scikit-learn over varying lambda.

    inputs:
        data (Data): Data for regression analysis
        kfolds (int): Number of bootstraps
        model (Model): Regression model to apply
        maxDim (int): Maximal polynomial degree
        lmbds (np.array): Regularization weights
    returns:
        (tuple[np.array, np.array]) of (Error, Variance)
    """
    polyDegrees = range(1, maxDim + 1)

    n_degrees = len(polyDegrees)
    n_lmbds = len(lmbds)
    error = np.zeros((n_degrees, n_lmbds))
    variance = np.zeros((n_degrees, n_lmbds))

    dummy_model = Model()  # only needed because of where create X is
    Kfold = skKfold(n_splits=kfolds, shuffle=True)

    pbar = tqdm(
        total=n_degrees * n_lmbds, desc=f"sklearn CV {model.__class__.__name__}"
    )
    for i, degree in enumerate(polyDegrees):
        X = dummy_model.create_X(data.x_, data.y_, degree)
        X = StandardScaler().fit_transform(X)
        for j, lambd in enumerate(lmbds):
            model.alpha = lambd

            scores = cross_val_score(
                model,
                X,
                data.z_,
                scoring="neg_mean_squared_error",
                cv=Kfold,
                n_jobs=-1,
            )
            error[i, j] = -scores.mean()
            variance[i, j] = scores.std()
            pbar.update(1)

    return error, variance


def HomeMade_cross_val_lambdas(
    data: Data,
    kfolds: int = 5,
    model: Ridge | Lasso = Ridge(),
    maxDim: int = 15,
    lmbds: np.array = np.logspace(-3, 5, 10),
) -> tuple[np.array, np.array]:
    """Perform cross validation over varying lambda.

    inputs:
        data (Data): Data for regression analysis
        kfolds (int): Number of bootstraps
        model (Model): Regression model to apply
        maxDim (int): Maximal polynomial degree
        lmbds (np.array): Regularization weights
    returns:
        (tuple[np.array, np.array]) of (Error, Variance)
    """
    polyDegrees = range(1, maxDim + 1)
    n_degrees = len(polyDegrees)
    n_lmbds = lmbds.size

    error = np.zeros((n_degrees, n_lmbds))
    variance = np.zeros((n_degrees, n_lmbds))

    Kfold = KFold(n_splits=kfolds, shuffle=True)

    pbar = tqdm(
        total=n_degrees * n_lmbds * kfolds,
        desc=f"Homemade CV {model.__class__.__name__}",
    )
    for i, degree in enumerate(polyDegrees):
        X = model.create_X(data.x_, data.y_, degree)
        X = StandardScaler().fit_transform(X)

        for j, lambd in enumerate(lmbds):
            scores = np.zeros(kfolds)
            for k, (train_i, test_i) in enumerate(Kfold.split(X)):
                X_train = X[train_i]
                X_test = X[test_i]
                z_test = data.z_[test_i]
                z_train = data.z_[train_i]

                model.fit(X_train, z_train, lambd)

                z_pred = model.predict(X_test)

                scores[k] = MSE(z_pred, z_test)
                pbar.update(1)

            error[i, j] = scores.mean()
            variance[i, j] = scores.std()
    return error, variance
