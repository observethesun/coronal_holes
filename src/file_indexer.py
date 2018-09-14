"""Contains FileIndexer class."""
import os
from os.path import basename
import glob
import numpy as np


def gen_indices(pathes, make):
    """Generate indices from filenames.

    Parameters
    ----------
    pathes : list
        File pathes.
    make: callable or None
        Method to generate an index from the filename. If None the filename is an index.

    Returns
    -------
    result : list
        List of indices.
    """
    if make is None:
        res = np.array([basename(os.path.split(f)[1]) for f in pathes])
    elif callable(make):
        res = np.array([make(os.path.split(f)[1]) for f in pathes])
    else:
        raise NotImplemented()

    if len(res) == len(set(res)):
        return res
    else:
        ind, count = np.unique(res, return_counts=True)
        raise ValueError('Indices are not unique: ', ind[count > 1])


class FileIndexer():
    """Indexing if files.

    Attributes
    ----------
    indices : ndarray
        Unique identifiers for items in the dataset.
    masks : ndarray
        Pathes to masks.
    images : ndarray
        Pathes to files.
    batch_class : class
        Output batch class for generator
    train : FileIndexer or None
        File indexer for train part of the dataset
    test : FileIndexer or None
        File indexer for test part of the dataset
    """
    def __init__(self, components=None, method=None):
        self.indices = None
        self.images = None
        self.masks = None
        self.train = None
        self.test = None

        if components is None:
            components = []

        for component in components:
            pathes = np.array(glob.glob(component['path']))
            if 'make_indices' in component:
                indices = gen_indices(pathes, component['make_indices'])
                order = np.argsort(indices)
                indices = indices[order]
            else:
                indices = gen_indices(pathes, None)
                order = np.argsort(pathes)

            if self.indices is None:
                self.indices = indices
                setattr(self, component['name'], pathes[order])
            else:
                if np.array_equal(self.indices, indices):
                    setattr(self, component['name'], pathes[order])
                else:
                    print('missed masks:', set(self.indices) - set(indices))
                    print('missed images:', set(indices) - set(self.indices))
                    raise ValueError('Failed to match masks and images')

    def __len__(self):
        return len(self.indices)

    def _from_arrays(self, indices, components=None):
        """Set indices and components."""
        self.indices = indices
        if components is None:
            components = []
        for component in components:
            setattr(self, component['name'], component['path'])
        return self

    def subset(self, pos):
        """Get FileIndexer on a subset of the dataset.

        Parameters
        ----------
        pos : list
            Index positions to form a subset.

        Returns
        -------
        result : FileIndexer
            FileIndexer on a subset.
        """
        components = []
        for component in ['images', 'masks']:
            if getattr(self, component) is not None:
                components.append({'name': component, 'path': getattr(self, component)[pos]})
        return FileIndexer()._from_arrays(self.indices[pos], components)

    def train_test_split(self, train_ratio=0.8, suffle=False, seed=None):
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
        result : FileIndexer
            FileIndexer with train and test subsets.
        """
        order = np.arange(len(self))
        if suffle:
            np.random.seed(seed)
            np.random.shuffle(order)
        train_size = int(train_ratio * len(self))
        self.train = self.subset(order[:train_size])
        self.test = self.subset(order[train_size:])
        return self

    def shuffle(self, seed=None):
        """Randomly shuffle indices.

        Parameters
        ----------
        seed : int
            Seed for Numpy RandomState.

        Returns
        -------
        result : FileIndexer
            FileIndexer with shuffled indices.
        """
        order = np.arange(len(self))
        np.random.seed(seed)
        np.random.shuffle(order)
        self = self.subset(order)
        return self
