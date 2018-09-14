"""Contains ImageBatch class."""
import os
import numpy as np
from scipy.misc import imread
import dill
import blosc

from .decorators import threads
from .file_indexer import FileIndexer


def get_objects(path, types=None):
    """Read pixel coordinates of objects from abp file.

    Parameters
    ----------
    path : str
        The path to abp file.
    types : list or None
        If None read all objects. Else read object types given in list.
        Available types are "spots" and "cores". Default to None.

    Returns
    -------
    object_list : ndarray
        Array that containes arrays of pixel coordinates for each type of objects.
    """
    spots = []
    cores = []
    with open(path, 'r') as fin:
        fread = fin.readlines()
        num_objects = int(fread[2])
        for i in range(num_objects):
            info = np.array(fread[3 + 2 * i].split()).astype('int')
            if (info[-1] == 0) and (types is None):
                data = fread[4 + 2 * i].split()
                spots.extend(data)
            elif (info[-1] == 0) and ('spots' in types):
                data = fread[4 + 2 * i].split()
                spots.extend(data)
            elif (info[-1] != 0) and ('cores' in types):
                data = fread[4 + 2 * i].split()
                cores.extend(data)
            else:
                continue

    object_list = []
    if types is None:
        spots = np.array(spots).reshape((-1, 3))[:, 1:None:-1].astype('int')
        object_list.append(spots)
    else:
        if 'spots' in types:
            spots = np.array(spots).reshape((-1, 3))[:, 1:None:-1].astype('int')
            object_list.append(spots)
        if 'cores' in types:
            cores = np.array(cores).reshape((-1, 3))[:, 1:None:-1].astype('int')
            object_list.append(cores)

    return np.array(object_list + [None])[:-1]


def downsize_img(arr, target_size, average=None):
    """Reduce image size to target size.

    Parameters
    ----------
    arr : ndarray
        Array to be resized.
    target_size : int
        Target size of resized image.
    average : callable or None
        Method to use for image reduction. If None then np.mean averaging is applied.
        Default to None.

    Returns
    -------
    img : ndarray
        Resized image.
    """
    if arr.shape[0] % target_size != 0:
        raise ValueError("Image size is not divisible by target size")
    if arr.shape[0] == target_size:
        return arr
    if average is None:
        average = np.mean
    factor = int(arr.shape[0] / target_size)
    shape = (target_size, target_size, factor, factor) + arr.shape[2:]
    strides = (arr.strides[0] * factor, arr.strides[1] * factor, arr.strides[0], arr.strides[1], ) + arr.strides[2:]
    return average(np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides), axis=(2, 3))


class ImageBatch():
    """Batch class for images and masks processing.

    Attributes
    ----------
    index : FileIndex
        Unique identifiers of batch items.
    images : ndarray
        Array of images.
    masks : ndarray
        Array of segmentation masks.
    headers : ndarray
        Array of positions of solar disk and other metadata.
    """
    def __init__(self, index, **kwargs):
        self.index = index
        self.images = np.array([np.array([]) for _ in range(len(index))] + [None])[:-1]
        self.masks = np.array([np.array([]) for _ in range(len(index))] + [None])[:-1]
        self.headers = np.array([np.array([]) for _ in range(len(index))] + [None])[:-1]

    @property
    def indices(self):
        return self.index.indices

    def __len__(self):
        return len(self.index)

    def deepcopy(self):
        """Make a deep copy of batch"""
        batch_dump = dill.dumps(self)
        return dill.loads(batch_dump)

    def _apply_to_scope(self, method, scope, index, *args, **kwargs):
        """Apply method to batch items by scope and index"""
        for component in scope:
            attr = getattr(self, component)
            attr[index] = method(attr[index], *args, **kwargs)

    @threads()
    def apply(self, index, method, scope, *args, **kwargs):
        """Apply given method to batch components from scope.

        Parameters
        ----------
        method : callable
            Function to be applied to items of batch component.
            First argument should be an item of batch component.
        scope : list
            List of components to which the method should be applied.

        Returns
        -------
        batch : ImageBatch
            Batch with modified components.
        """
        self._apply_to_scope(method, scope, index, *args, **kwargs)

    @threads()
    def load_images(self, index, fmt, dtype=None, **kwargs):
        """Load images from files.

        Parameters
        ----------
        fmt : str
            Format of image files.
        dtype: str
            Cast images to specified dtype.

        Returns
        -------
        batch : ImageBatch
            Batch with uploaded images.
        """
        fname = self.index.images[index]
        if fmt == 'blosc':
            with open(fname, 'rb') as f:
                img = dill.loads(blosc.decompress(f.read()))
        elif fmt == 'jpeg':
            img = imread(fname, mode='L')
        else:
            raise ValueError('Unknown file format')
        if dtype:
            img = img.astype(dtype)
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        origin = np.array((((0, 0), (0, img.shape[1])), ((img.shape[0], 0), (img.shape[0], img.shape[1]))))
        self.images[index] = img

    @threads()
    def dump_images(self, index, fmt, path):
        """Dump images.

        Returns
        -------
        batch : ImageBatch
            Batch unchanged.
        """
        fname = os.path.join(path, str(self.indices[index]))
        data = self.images[index]
        if fmt == 'blosc':
            with open(fname + '.blosc', 'w+b') as f:
                f.write(blosc.compress(dill.dumps(data)))
        else:
            raise ValueError('Unknown file format')

    @threads()
    def load_headers(self, index):
        """Load headers from abp files.

        Returns
        -------
        batch : ImageBatch
            Batch with uploaded headers.
        """
        with open(self.index.masks[index], 'r') as fin:
            fread = fin.readlines()
            header = np.array(fread[0].split()).astype('float')
        self.headers[index] = header

    @threads()
    def load_objects(self, index, types=None):
        """Load objects from abp files.

        Returns
        -------
        batch : ImageBatch
            Batch with uploaded objects.
        """
        self.masks[index] = get_objects(self.index.masks[index], types)

    @threads()
    def make_segmentation_masks(self, index, shape, ohe=False):
        """Make segmentation mask according to pixel coordinates of objects.

        Parameters
        ----------
        shape : tuple
            Shape of target segmentation mask.
        ohe: bool
            If true mask is one-hot-encoded for each type of objects and for background.
            Default to false.

        Returns
        -------
        batch : ImageBatch
            Batch with segmentation masks.
        """
        objects = self.masks[index]
        if ohe:
            mask = np.zeros(shape + (len(objects) + 1,), dtype='int8')
            mask[:, :, 0] = 1
            for i, obj in enumerate(objects):
                pattern = [0] * (i + 1) + [1]
                mask[obj[:, 0], obj[:, 1], :i + 2] = pattern
        else:
            mask = np.zeros(shape + (1,), dtype='int8')
            for i, obj in enumerate(objects):
                mask[obj[:, 0], obj[:, 1]] = i + 1
        self.masks[index] = mask

    @threads()
    def mask_disk(self, index, value=255):
        """Set the same value to pixels outside the solar disk.

        Parameters
        ----------
        value : int or float
            A value to be set to pixels. Default to 255.

        Returns
        -------
        batch : ImageBatch
            Batch with modified images.
        """
        img = self.images[index]
        xc, yc, r = self.headers[index][:3].astype(int)
        ind = np.transpose(np.indices(img.shape[:2]), axes=(1, 2, 0))
        img[np.where(np.linalg.norm(ind - [xc, yc], axis=-1) > r)] = value

    def drop_empty_days(self):
        """Drop images without objects.

        Returns
        -------
        batch : ImageBatch
            Batch without empty days.
        """
        valid_days = [i for i in range(len(self)) if np.any([len(x) > 0 for x in self.masks[i]])]
        self.images = self.images[valid_days]
        self.masks = self.masks[valid_days]
        self.headers = self.headers[valid_days]
        self.index = self.index.subset(valid_days)
        return self

    @threads()
    def downsize_image(self, index, scope, target_size):
        """Reduce image size to traget size.

        Parameters
        ----------
        scope : list
            List of components to which the method should be applied.
        target_size : int
            Target size of resized image.

        Returns
        -------
        batch : ImageBatch
            Batch with downsized images.
        """
        self._apply_to_scope(downsize_img, scope, index, target_size)

    @threads(random_kwarg={'kwarg': 'k', 'generator': lambda size: np.random.choice([True, False], size=size)})
    def random_flip(self, index, scope, k, axis):
        """Apply with probability 0.5 reversing of the order of entries in an image
        along the given axis.

        Parameters
        ----------
        axis : int
            Axis in array, which entries are reversed.

        Returns
        -------
        batch : ImageBatch
            Batch with flipped images.
        """
        if k:
            self._apply_to_scope(np.flip, scope, index, axis)

    @threads(random_kwarg={'kwarg': 'k', 'generator': lambda size: np.random.choice([0, 1, 2, 3], size=size)})
    def random_rot90(self, index, scope, k, axes=(-3, -2)):
        """Apply rotation of an image at random number of times by 90 degrees.

        Parameters
        ----------
        axes: (2,) array_like
            The array is rotated in the plane defined by the axes. Axes must be different.

        Returns
        -------
        batch : ImageBatch
            Batch with rotated images.
        """
        self._apply_to_scope(np.rot90, scope, index, k, axes)
