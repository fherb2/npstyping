import numpy as np
from stype import SType, STypeLike, Colon
from typing import Union
from numpy.typing import ArrayLike
from types import EllipsisType


# ############################################################
#
# sndarray (shape typed numpy.ndarray)
# ====================================
#
# Is a subclass of numpy.ndarray with additional:
#
#   -   stype   – Attribut. Saves a SType value as shape restriction
#                 parameter to can be cecked every time against the
#                 array of this numpy.ndarray subclass.
#
#   -   auto_stype_check – Attribute. If true, so an shape check
#                          happens after each numpy operation.
#
#   -   Mechanism to keep the stype attribute in such cases where
#       a numpy operation creates a new ndarray.
#
#   -   check_stype – Method to check the restriction of the shape
#                     against the current array shape


class sndarray(np.ndarray):
    
    def __new__(cls, array_like, dtype:np.dtype = None,
                stype:SType = None, auto_stype_check:bool = False,
                order = None, like=None):
        obj = np.asarray(array_like, dtype=dtype, order=order, like=like).view(cls)
        obj._stype = SType(stype)
        assert isinstance(auto_stype_check, bool | None), ValueError("Argument 'auto_stype_check' has wrong data type (not a boolean).")
        obj._auto_stype_check = auto_stype_check
            
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None:
            return None
        self.stype = getattr(obj, '_stype', None) # we use the setter method to get
                                                  # the boolean-True behaviour
        self._auto_stype_check = getattr(obj, '_auto_stype_check', None)
        
    @property
    def stype(self):
        return self._stype
    
    @stype.setter
    def stype(self, stype_like: STypeLike | bool):
        if isinstance(stype_like, bool):
            if stype_like:
                # its boolean 'True'; means: Take array's current shape as stype.
                self._stype = self.shape
        else:
            # should be an stype_like shape contraint
            self._stype = SType(stype_like)
            
    @property
    def auto_stype_check(self):
        return self._auto_stype_check
    
    @auto_stype_check.setter
    def auto_stype_check(self, auto_stype_check:bool):
        assert isinstance(auto_stype_check, bool | None), ValueError("Argument 'auto_stype_check' has wrong data type (not a boolean).")
        self._auto_stype_check = auto_stype_check
            
    def check_stype(self, stype_like: STypeLike | None = None, auto_stype_check:bool = False):
        if stype_like is not None:
            self.stype = stype_like
            assert isinstance(auto_stype_check, bool | None), ValueError("Argument 'auto_stype_check' has wrong data type (not a boolean).")
            self._auto_stype_check = auto_stype_check
    
    #
    # parts of code to implement 'auto-check' and 'keep stype' behaviour
    # ------------------------------------------------------------------
    #

    # implement the process into all of numpy operations
    def __getattribute__(self, name):
        attr = super().__getattribute__(name)

        if callable(attr):
            # an dieser Stelle könnte es noch einen Iterationsfehler geben: Darf nicht beim Erzeugen
            # oder initialisieren der Klasse aufgerufen werden.
            def wrapper_method(*args, **kwargs):
                result = attr(*args, **kwargs)
                if isinstance(result, np.ndarray):
                    result = sndarray(result, dtype=result.dtype,
                                      stype=self._stype, auto_stype_check=self._auto_stype_check)
                if self._auto_stype_check:
                    result.check_stype()

                return result
        
            return wrapper_method
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
    
