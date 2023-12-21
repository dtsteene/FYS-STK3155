import numpy as np


class KFold:
    def __init__(self, n_splits: int, shuffle: bool = True) -> None:
        """Initialize KFold

        inputs:
            n_splits (int): Number of folds
            shuffle (bool): Whether to shuffle the indices
        """
        self.n_splits = n_splits
        self.shuffle = shuffle

    def split(self, X: np.array) -> list[tuple[np.array, np.array]]:
        """Split the working array into equal sized folds.

        inputs:
            X (np.array): Array to split
        returns:
            List of training and testing indicies
        """
        length = X.shape[0]
        indices = np.arange(length)

        if self.shuffle:
            np.random.shuffle(indices)

        results = []
        kfoldLength = length // self.n_splits
        for i in range(0, self.n_splits):
            # left = indices[:]
            training = np.concatenate(
                (indices[: i * kfoldLength], indices[(i + 1) * kfoldLength :])
            )
            testing = indices[i * kfoldLength : (i + 1) * kfoldLength]
            results += [(training, testing)]

        return results


if __name__ == "__main__":
    kfold = KFold(5, shuffle=False)
    x = np.arange(25)
    for training, testing in kfold.split(x):
        print()
        print(x[training])
        print(x[testing])
        print()
