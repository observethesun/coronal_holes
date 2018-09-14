"""Contains BatchGenerator class."""
import numpy as np


class BatchGenerator():
    """Generation of batches over a dataset.

    Attributes
    ----------
    index : FileIndexer
        File indexer for dataset items.
    indices : ndarray
        Dataset indices.
    batch_class : class
        Output batch class for generator
    train : BatchGenerator or None
        Batch generator for train part of the dataset
    test : BatchGenerator or None
        Batch generator for test part of the dataset
    """
    def __init__(self, index, batch_class):
        self.index = index
        self.batch_class = batch_class
        self.train = None
        self.test = None
        self._batch_start = 0
        self._on_epoch = 0
        self._order = np.arange(len(index))

    @property
    def indices(self):
        return self.index.indices

    def __len__(self):
        return len(self.index)

    def reset(self):
        """Reset epochs counter to 0."""
        self._batch_start = 0
        self._on_epoch = 0
        return self

    def train_test_split(self, train_ratio=0.8, shuffle=False, seed=None):
        """Randomly splits the dataset into train and test subsets.

        Parameters
        ----------
        train_ratio : float, in [0, 1]
            Ratin of the train subset to the whole dataset. Default to 0.8.
        shuffle : bool
            Should the dataset be shuffled before splitting into train and test.
            Default to False
        seed : int
            Seed for Numpy RandomState.

        Returns
        -------
        result : BatchGenerator
            Batch generator with train and test subsets.
        """
        self.index.train_test_split(train_ratio, shuffle, seed)
        self.test = BatchGenerator(self.index.test, self.batch_class)
        self.train = BatchGenerator(self.index.train, self.batch_class)
        return self

    def shuffle(self, seed=None):
        """Randomly shuffle indices.

        Parameters
        ----------
        seed : int
            Seed for Numpy RandomState.

        Returns
        -------
        result : BatchGenerator
            Batch generator with shuffled indices.
        """
        self.index = self.index.shuffle(seed)
        return self

    def next_batch(self, batch_size, n_epochs=None, shuffle=False, drop_last=True, **kwargs):
        """Get next batch.

        Parameters
        ----------
        batch_size : int
            Size of target batch.
        n_epochs : int or None
            Maximal number of epochs. If None then next_batch can be
            called infinitely. Default to None.
        suffle : bool
            Should the dataset be suffled after each epoch. Default to False.
        drop_last : bool
            Defines handling of incomplete batches in the end of epochs.
            If drop_last is False we complete batch with items from the beginning
            of the current epoch or return incomplete batch if current epoch
            is the last epoch. If drop_last is True we go to the next epoch.

        Returns
        -------
        batch : batch_class
            Generated batch.
        """
        if (n_epochs is not None) and (self._on_epoch >= n_epochs):
            raise ValueError('dataset is over')
        if batch_size + self._batch_start <= len(self):
            a, b = self._batch_start, self._batch_start + batch_size
        else:
            if (n_epochs is not None) and (self._on_epoch == n_epochs - 1):
                self._on_epoch += 1
                if drop_last:
                    return self.next_batch(batch_size, n_epochs, shuffle, drop_last)
                else:
                    a, b = self._batch_start, len(self.indices)
            else:
                if drop_last:
                    self._on_epoch += 1
                    self._batch_start = 0
                    if shuffle:
                        self._order = np.random.permutation(len(self._order))
                    return self.next_batch(batch_size, n_epochs, shuffle, drop_last)
                else:
                    self._on_epoch += 1
                    a, b = self._batch_start, (self._batch_start + batch_size) % len(self.indices)
                    next_items = np.hstack([self._order[a:], self._order[:b]])
                    self._batch_start = b
                    if shuffle:
                        self._order = np.random.permutation(len(self._order))
                    return self.batch_class(self.index.subset(next_items), **kwargs)

        next_items = self._order[a: b]
        if b == len(self):
            self._batch_start = 0
            self._on_epoch += 1
        else:
            self._batch_start = b

        return self.batch_class(self.index.subset(next_items), **kwargs)
