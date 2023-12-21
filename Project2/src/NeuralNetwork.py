import jax.numpy as np
import numpy as onp
from jax import grad, jit, lax
from typing import Optional, Callable
from Activators import sigmoid, derivate
from CostFuncs import CostOLS_fast
from Schedules import Scheduler
from sklearn.utils import resample
from copy import deepcopy
from tqdm import tqdm
from utils import assign, vstack_arrs
from functools import partial


@jit
def fast_mul(a, b):
    # Fast multiplication
    return lax.mul(a, b)


@jit
def fast_dot(a, b):
    # Fast dot product
    return lax.dot(a, b)


@jit
def fast_dot_with_T(a, b):
    # Fast dot product with transpose
    return lax.dot(a, b.T)


@jit
def fast_plus_eq(a, b):
    # Fast inplace addition
    return lax.add(a, b)


@partial(jit, static_argnames="lmbda")
def calc_grad_w(a_layer, delta_matrix, weight_matrix, lmbda):
    """Calculate the gradient for the weights of a layer.

    Calculates:
        a_layer.T @ delta_matrix + weight_matrix * lmbda

    Args:
        a_layer (np.ndarray): The activation layer.
        delta_matrix (np.ndarray): The delta matrix.
        weight_matrix (np.ndarray): The weight matrix.
        lmbda (float): The regularization parameter.

    Returns:
        np.ndarray: The gradient for the weights of a layer.
    """
    gradient_weights = lax.dot(a_layer.T, delta_matrix)
    gradient_weights = lax.add(gradient_weights, lax.mul(weight_matrix, lmbda))
    return gradient_weights


@jit
def setup_bias(X_batch: np.ndarray) -> np.ndarray:
    # Initialize and stack the bias
    bias = lax.mul(np.ones((X_batch.shape[0], 1)), 0.01)
    return np.hstack([bias, X_batch])


class NeuralNet:
    """
    A class representing a neural network.

    Attributes:
        dimensions : tuple[int]
            A tuple of integers representing the number of nodes in each layer of the neural network.
        hidden_func : Callable
            A Callable function representing the activation function for the hidden layers.
        output_func : Callable
            A Callable function representing the activation function for the output layer.
        cost_func : Callable
            A Callable function representing the cost function used to evaluate the performance of the neural network.
        seed : Optional[int]
            An optional integer representing the seed for the random number generator used to initialize the weights.

    Methods:
        reset_weights() -> None:
            Resets the weights of the neural network.
        fit(X_train: np.ndarray, target_train: np.ndarray, scheduler: Scheduler, **kwargs) -> dict[str, np.ndarray]:
            Trains the neural network on the given data.
        feed_forward(X_batch: np.ndarray) -> np.ndarray:
            Performs a feed forward pass through the neural network.
        back_propagate(X_batch: np.ndarray, target_batch: np.ndarray, lmbda: float) -> None:
            Performs back propagation to update the weights of the neural network.
        accuracy(prediction: np.ndarray, target: np.ndarray) -> float:
            Calculates the accuracy of the neural network.
        set_classification() -> None:
            Sets the classification attribute of the neural network based on the cost function used.
    """

    def __init__(
        self,
        dimensions: tuple[int],
        hidden_func: Callable = sigmoid,
        output_func: Callable = sigmoid,
        cost_func: Callable = CostOLS_fast,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initializes the neural network.

        Args:
            dimensions : tuple[int]
                A tuple of integers representing the number of nodes in each layer of the neural network.
            hidden_func : Callable
                A Callable function representing the activation function for the hidden layers.
            output_func : Callable
                A Callable function representing the activation function for the output layer.
            cost_func : Callable
                A Callable function representing the cost function used to evaluate
                the performance of the neural network.
            seed : Optional[int]
                An optional integer representing the seed for the random number generator
                used to initialize the weights.

        Raises:
            TypeError:
                If dimensions is not a tuple, if any value in dimensions is not an integer,
                or if seed is not an integer or None.
            ValueError:
                If any value in dimensions is less than or equal to 0.
        """
        if not isinstance(dimensions, tuple):
            raise TypeError(f"Dimensions must be tuple, not {type(dimensions)}")
        if not all(isinstance(layer, int) for layer in dimensions):
            raise TypeError(f"Values of dimensions must be ints, not {dimensions}")
        if not isinstance(seed, int) or seed is None:
            raise TypeError(f"Seed must be either None or int, not {type(seed)}")
        if any(dimension <= 0 for dimension in dimensions):
            raise ValueError(f"Number of dimensions must be positive, not {dimensions}")

        self.dimensions = dimensions
        self.hidden_func = jit(hidden_func)
        self.output_func = jit(output_func)
        self.cost_func = jit(cost_func)

        # Calculate derivates here to same on jit
        self.cost_func_derivative = jit(grad(self.cost_func))
        self.hidden_derivative = derivate(self.hidden_func)
        self.output_derivative = derivate(self.output_func)

        self.seed = seed

        self.z_layers: list[np.ndarray] = list()
        self.a_layers: list[np.ndarray] = list()

        self.reset_weights()
        self.set_classification()

    def reset_weights(self) -> None:
        """
        Resets the weights of the neural network.
        """
        if self.seed is not None:
            onp.random.seed(self.seed)

        self.weights = list()
        for i in range(len(self.dimensions) - 1):
            # Weights
            weight_array = onp.random.randn(
                self.dimensions[i] + 1, self.dimensions[i + 1]
            )

            # Bias
            weight_array[0, :] = onp.random.randn(self.dimensions[i + 1]) * 0.01

            self.weights.append(weight_array)

    def fit(
        self,
        X_train: np.ndarray,
        target_train: np.ndarray,
        scheduler: Scheduler,
        batches: int = 1,
        epochs: int = 100,
        lmbda: float = 0.0,
        X_val: Optional[np.ndarray] = None,
        target_val: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        """
        Trains the neural network on the given data.

        Args:
            X_train : np.ndarray
                A numpy array representing the training data.
            target_train : np.ndarray
                A numpy array representing the target values for the training data.
            scheduler : Scheduler
                A scheduler object used to update the weights of the neural network.
            batches : int, optional
                An integer representing the number of batches to divide the training data into, by default 1.
            epochs : int, optional
                An integer representing the number of epochs to train the neural network for, by default 100.
            lmbda : float, optional
                A float representing the regularization parameter, by default 0.
            X_val : Optional[np.ndarray], optional
                An optional numpy array representing the validation data, by default None.
            target_val : Optional[np.ndarray], optional
                An optional numpy array representing the target values for the validation data, by default None.

        Returns:
            dict[str, np.ndarray]
                A dictionary containing the training and validation errors and accuracies (if applicable).

        Raises:
            TypeError:
                If scheduler is not of class Scheduler, if batches or epochs are not integers,
                or if lmbda is not a number.
            ValueError:
                If batches or epochs are less than or equal to 0, if lmbda is negative, or if
                the number of batches exceeds the number of training points.
        """
        # Handle TypeErrors (arrays are iffy with jax etc.)
        if not isinstance(scheduler, Scheduler):
            raise TypeError("The scheduler must be of class Scheduler")
        if not isinstance(batches, int):
            raise TypeError(f"Number of batches must be int, not {type(batches)}")
        if not isinstance(epochs, int):
            raise TypeError(f"Number of epochs must be int, not {type(epochs)}")

        # Handle ValueErrors
        if batches <= 0:
            raise ValueError(f"Number of batches must be positive, not {batches}")
        if epochs <= 0:
            raise ValueError(f"Number of epochs must be positive, not {epochs}")
        if lmbda < 0:
            raise ValueError(f"lmbda cannot be negative, {lmbda=}")

        # Set random seed to exclude source of error
        if self.seed is not None:
            onp.random.seed(self.seed)

        # Validate if available
        validate = False
        if X_val is not None and target_val is not None:
            validate = True

        # Cast to float for jax
        lmbda = float(lmbda)

        # Training metrics
        train_errors = np.empty(epochs)
        train_accuracies = np.empty(epochs)

        self.schedulers_weight: list[Scheduler] = list()
        self.schedulers_bias: list[Scheduler] = list()

        if X_train.shape[0] < batches:
            raise ValueError(
                f"Number of batches cannot exceed training points, {X_train.shape[0]} < {batches}"
            )

        batch_size = X_train.shape[0] // batches

        # One step of bootstrap
        X_train, target_train = resample(X_train, target_train)

        if validate:
            # Validation metrics
            validation_errors = np.empty(epochs)
            validation_accuracies = np.empty(epochs)

        for i in range(len(self.weights)):
            # I believe deepcopy is necessary to ensure layers are not cross contaminated
            self.schedulers_weight.append(deepcopy(scheduler))
            self.schedulers_bias.append(deepcopy(scheduler))

        data_indices = np.arange(X_train.shape[0])

        pbar = tqdm(total=epochs * batches)
        for e in range(epochs):
            for i in range(batches):
                # Draw with replacement
                batch_idx = onp.random.choice(data_indices, batch_size)
                X_batch = X_train[batch_idx, :]
                target_batch = target_train[batch_idx]

                self.feed_forward(X_batch)
                self.back_propagate(X_batch, target_batch, lmbda)

                pbar.update(1)

            # Reset schedulers
            for weight_scheduler, bias_scheduler in zip(
                self.schedulers_weight, self.schedulers_bias
            ):
                weight_scheduler.reset()
                bias_scheduler.reset()

            pred_train = self.predict(X_train)
            train_error = self.cost_func(pred_train, target_train)
            train_errors = assign(train_errors, e, train_error)

            if validate:
                pred_val = self.predict(X_val)
                validation_error = self.cost_func(pred_val, target_val)
                validation_errors = assign(validation_errors, e, validation_error)

            if self.classification:
                train_accuracy = self.accuracy(pred_train, target_train)
                train_accuracies = assign(train_accuracies, e, train_accuracy)

            if validate and self.classification:
                pred_accuracy = self.accuracy(pred_val, target_val)
                validation_accuracies = assign(validation_accuracies, e, pred_accuracy)

        scores = {"train_errors": train_errors}
        if validate:
            scores["validation_errors"] = validation_errors
        if self.classification:
            scores["train_accuracy"] = train_accuracies
        if validate and self.classification:
            scores["validation_accuracy"] = validation_accuracies

        return scores

    def feed_forward(self, X_batch: np.ndarray) -> np.ndarray:
        """
        Performs a feed forward pass through the neural network.

        Args:
            X_batch : np.ndarray
                A numpy array representing the input data.

        Returns:
            np.ndarray
                A numpy array representing the output of the neural network.
        """
        self.a_layers = list()
        self.z_layers = list()

        # Make sure X is a matrix
        if len(X_batch.shape) == 1:
            X_batch = X_batch.reshape((1, X_batch.size))

        # Add a bias
        X_batch = setup_bias(X_batch)

        a = X_batch
        self.a_layers.append(a)
        self.z_layers.append(a)

        # Feed forward for all but output layer
        for i in range(len(self.weights) - 1):
            z = fast_dot(a, self.weights[i])
            self.z_layers.append(z)
            a = self.hidden_func(z)

            # Add bias layer
            a = setup_bias(a)
            self.a_layers.append(a)

        # Output layer
        z = fast_dot(a, self.weights[-1])
        a = self.output_func(z)

        self.a_layers.append(a)
        self.z_layers.append(z)

        # Return the output layer
        return a

    def back_propagate(
        self, X_batch: np.ndarray, target_batch: np.ndarray, lmbda: float
    ) -> None:
        """
        Performs back propagation to update the weights of the neural network.

        Args:
            X_batch : np.ndarray
                A numpy array representing the input data.
            target_batch : np.ndarray
                A numpy array representing the target values for the input data.
            lmbda : float
                A float representing the regularization parameter.

        Raises:
            ValueError:
                If the shapes of prediction and target do not correspond.
        """

        # Start with output layer
        i = len(self.weights) - 1

        if self.output_func.__name__ == "softmax":
            delta_matrix = self.a_layers[i + 1] - target_batch
        else:
            left = self.output_derivative(self.z_layers[i + 1])
            right = self.cost_func_derivative(self.a_layers[i + 1], target_batch)
            delta_matrix = fast_mul(left, right)

        # Output gradient
        gradient_bias = np.sum(delta_matrix, axis=0).reshape(1, delta_matrix.shape[1])
        gradient_weights = calc_grad_w(
            self.a_layers[i][:, 1:], delta_matrix, self.weights[i][1:, :], lmbda
        )

        update_matrix = vstack_arrs(
            self.schedulers_bias[i].update_change(gradient_bias),
            self.schedulers_weight[i].update_change(gradient_weights),
        )

        self.weights[i] -= update_matrix

        # Back propagate the hidden layers
        for i in range(len(self.weights) - 2, -1, -1):
            # Calculate error for layer
            left = fast_dot_with_T(self.weights[i + 1][1:, :], delta_matrix)
            right = self.hidden_derivative(self.z_layers[i + 1])

            delta_matrix = fast_mul(left.T, right)

            # Calculate gradients
            gradient_bias = np.sum(delta_matrix, axis=0).reshape(
                1, delta_matrix.shape[1]
            )
            gradient_weights = calc_grad_w(
                self.a_layers[i][:, 1:], delta_matrix, self.weights[i][1:, :], lmbda
            )

            update_matrix = vstack_arrs(
                self.schedulers_bias[i].update_change(gradient_bias),
                self.schedulers_weight[i].update_change(gradient_weights),
            )
            # Update weights
            self.weights[i] -= update_matrix

    def accuracy(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """
        Calculates the accuracy of the neural network.

        Args:
            prediction : np.ndarray
                A numpy array representing the predicted values.
            target : np.ndarray
                A numpy array representing the target values.

        Returns:
            float
                A float representing the accuracy of the neural network.

        Raises:
            ValueError:
                If the shapes of prediction and target do not correspond.
        """
        if prediction.shape != target.shape:
            raise ValueError(
                f"Shapes must correspond, not {prediction.shape} and {target.shape}"
            )
        return np.average((target == prediction))

    def set_classification(self) -> None:
        """
        Sets the classification attribute of the neural network based on the cost function used.
        """
        self.classification = self.cost_func.__name__ in [
            "CostLogReg",
            "CostCrossEntropy",
            "CostCrossEntropy_binary",
        ]

    def predict(self, X: np.ndarray, *, theshold: float = 0.5) -> np.ndarray:
        """
        Predicts the output for the given input data.

        Args:
            X (np.ndarray): The input data to predict the output for.
            theshold (float, optional): The threshold value for classification. Defaults to 0.5.

        Returns:
            np.ndarray: The predicted output for the given input data.
        """
        predict = self.feed_forward(X)
        if self.classification:
            return np.where(predict > theshold, 1.0, 0.0)
        return predict
