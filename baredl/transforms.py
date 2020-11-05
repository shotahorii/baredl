from abc import ABCMeta, abstractmethod
import numpy as np
from .core import as_tensor
from .utils import pair

# PIL is not in dependency list
try:
    from PIL import Image
except ImportError:
    Image = None


class Compose:
    """ 
    Compose multiple transforms 
    
    Parameters
    ----------
    transforms: list of Transform instances
    """
    
    def __init__(self, transforms=[]):
        self.transforms = transforms

    def __call__(self, data):
        """
        Apply transforms

        Parameters
        ----------
        data: ndarray (any shape)
        """
        if not self.transforms:
            return data
        
        for f in self.transforms:
            data = f(data)
        return data


class Transform(metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, obj):
        """
        Parameters
        ----------
        obj: np.ndarray or PIL image
        """
        pass


class TransformPIL(Transform):

    def __init__(self):
        if Image is None:
            raise Exception('Image is not available. Install PIL.')


# -------------------------------------------------------------
# Transforms for np.ndarray
# -------------------------------------------------------------


class Flatten(Transform):
    def __call__(self, array):
        return array.flatten()


class AsType(Transform):
    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, array):
        return array.astype(self.dtype)


class ToInt(AsType):
    def __init__(self):
        self.dtype = np.int


class ToFloat(AsType):
    def __init__(self):
        self.dtype = np.float32


class ToTensor(Transform):
    def __call__(self, array):
        return as_tensor(array)


class Normalise(Transform):
    """Normalize a NumPy array with mean and standard deviation.
    Args:
        mean (float or sequence): mean for all values or sequence of means for
         each channel.
        std (float or sequence):
    """
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, array):
        mean, std = self.mean, self.std

        if not np.isscalar(mean):
            mshape = [1] * array.ndim
            mshape[0] = len(array) if len(self.mean) == 1 else len(self.mean)
            mean = np.array(self.mean, dtype=array.dtype).reshape(*mshape)
        if not np.isscalar(std):
            rshape = [1] * array.ndim
            rshape[0] = len(array) if len(self.std) == 1 else len(self.std)
            std = np.array(self.std, dtype=array.dtype).reshape(*rshape)
        return (array - mean) / std


# -------------------------------------------------------------
# Transforms for PIL Image
# -------------------------------------------------------------


class ToArray(TransformPIL):
    """Convert PIL Image to NumPy array."""
    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            return img
        if isinstance(img, Image.Image):
            img = np.asarray(img)
            img = img.transpose(2, 0, 1)
            img = img.astype(self.dtype)
            return img
        else:
            raise TypeError


class ToPIL(TransformPIL):
    """Convert NumPy array to PIL Image."""
    def __call__(self, array):
        data = array.transpose(1, 2, 0)
        return Image.fromarray(data)


class Resize(TransformPIL):
    """Resize the input PIL image to the given size.
    Args:
        size (int or (int, int)): Desired output size
        mode (int): Desired interpolation.
    """
    def __init__(self, size, mode=Image.BILINEAR):
        self.size = pair(size)
        self.mode = mode

    def __call__(self, img):
        return img.resize(self.size, self.mode)