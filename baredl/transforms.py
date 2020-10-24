from abc import ABCMeta

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
    pass