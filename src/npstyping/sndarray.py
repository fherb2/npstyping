import numpy as np
from stype import SType, ColonType
from typing import Union

class sndarray(np.ndarray):

    def __new__(cls, array_like, dtype:np.dtype = None, stype:SType = None, order = None, device=None, copy=None, like=None):
        obj = np.asarray(array_like, dtype=dtype, order=order, device=device, copy=copy, like=like)
        obj.stype = SType(stype)
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None:
            return None
        self.stype = getattr(obj, 'stype', None)

    @classmethod
    def _lazy_override(cls, method_name):
        """Overwrite a method by the first call"""
        original_method = getattr(np.ndarray, method_name)

        def wrapper(self, *args, **kwargs):
            """Generalized wrapper for all ndarray methodes, to convert ndarray into sndarray"""
            result:Union[np.ndarray | sndarray] = original_method(self, *args, **kwargs)

            # Is a new np.ndarray returned, instead of the original sndarray?
            if isinstance(result, np.ndarray) \
            and not isinstance(result, sndarray):
                # so we convert it to this subclass
                result = result.view(sndarray)

            # and overwrite the method permanently
            setattr(cls, method_name, original_method)
            return result
        
        # we catch the first call
        setattr(cls, method_name, wrapper)

    def __getattribute__(self, name):
        """Catches method calls and overwrite this methods on-the-fly."""
        try:
            attr = super().__getattribute__(name)
        except AttributeError:
            attr = getattr(np.ndarray, name, None)

        # overwrite only if this is method of np.ndarray
        if callable(attr) \
          and not name.startswith("_") \
          and not hasattr(self, name):
            type(self)._lazy_override(name)
            # return the new method
            return getattr(self, name)
        
        return attr