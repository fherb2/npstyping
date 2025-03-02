import numpy as np
from stype import SType, Colon
from typing import Union
from numpy.typing import ArrayLike
from types import EllipsisType

class sndarray(np.ndarray):
    SHAPE_CHANGING_METHODS = {
            "reshape", "resize", "transpose", "swapaxes",
            "flatten", "ravel", "squeeze", "expand_dims"
        }
    
    @classmethod
    def add_shape_changing_method(cls, method_name):
        cls.SHAPE_CHANGING_METHODS.add(method_name)

    @classmethod
    def remove_shape_changing_method(cls, method_name):    
        cls.SHAPE_CHANGING_METHODS.discard(method_name)

    def __new__(cls, array_like, dtype:np.dtype = None, stype:SType = None, order = None, like=None):
        obj = np.asarray(array_like, dtype=dtype, order=order, like=like).view(cls)
        obj.stype = SType(stype)
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None:
            return None
        self.stype = getattr(obj, 'stype', None)

    def post_process(self):
        print("POST-PROCESS")

    def __getattribute__(self, name):
        attr = super().__getattribute__(name)

        if callable(attr) and name in self.SHAPE_CHANGING_METHODS:

            def wrapper_method(*args, **kwargs):
                old_shape = self.shape 
                result = attr(*args, **kwargs)
                new_shape = result.shape if isinstance(result, np.ndarray) else None

                if new_shape and new_shape != old_shape:
                    if isinstance(result, sndarray):
                        result.post_process()
                else:
                    return result
            
            return wrapper_method
        else:
            return attr
    

arr = sndarray([1,2])
reshaped = arr.reshape(2,1)
print(isinstance(arr, sndarray))
print(isinstance(reshaped, sndarray))


    # @classmethod
    # def _lazy_override(cls, method_name): j73
    #     """Overwrite a method by the first call"""
    #     original_method = getattr(np.ndarray, method_name)

    #     def wrapper(self, *args, **kwargs):
    #         """Generalized wrapper for all ndarray methodes, to convert ndarray into sndarray"""
    #         result = original_method(self, *args, **kwargs)

    #         # Is a new np.ndarray returned, instead of the original sndarray?
    #         if isinstance(result, np.ndarray) \
    #         and not isinstance(result, sndarray):
    #             # so we convert it to this subclass
    #             result = result.view(sndarray)

    #         # and overwrite the method permanently
    #         setattr(cls, method_name, original_method)
    #         return result
        
    #     # we catch the first call
    #     setattr(cls, method_name, wrapper)

    # def __getattribute__(self, name):
    #     """Catches method calls and overwrite this methods on-the-fly."""
    #     try:
    #         attr = super().__getattribute__(name)
    #     except AttributeError:
    #         attr = getattr(np.ndarray, name, None)

    #     # overwrite only if this is method of np.ndarray
    #     if callable(attr) \
    #       and not name.startswith("_") \
    #       and not hasattr(self, name):
    #         type(self)._lazy_override(name)
    #         # return the new method
    #         return getattr(self, name)
        
    #     return attr
    
