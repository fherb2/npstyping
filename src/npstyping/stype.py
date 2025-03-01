import numpy as np

import re
from types import EllipsisType
from typing import Any, Literal
import numpy as np
from numpy.typing import ArrayLike
from enum import Enum
from dataclasses import dataclass

# ############################################################
#
# Common / simple type definitions, Constants
# ===============================
#

DO_TYPECHECK = False
if __debug__:
    DO_TYPECHECK = True

#
# Ending: Common / simple type definitions, Constants
#
# ############################################################

# ############################################################
#
# ColonType
# =========
#

#
# We use a helper "Meta" class in order to make isinstance(obj, ColonType)
# more duck_typing.
# Metaclass: _ColonType_Meta
# ----------------------
#

class _ColonType_Meta(type):
    
    @classmethod
    def __instancecheck__(cls, obj):
        if DO_TYPECHECK:
            # We check for the content, a single valid sign:
            #       >>   ":"   <<
            if type(obj) == _ColonType_Meta:
                return True
            elif obj == ":":
                return True
            elif obj == Literal[":"]:
                return True
            return False
        return True # DO_TYPECHECK == False
    
    # The most simple form of __call__(cls, *args, **kwargs) looks like:
    #
    #   if not args and not kwargs:
    #       return cls
    #   return super().__call__(*args, **kwargs)
    #
    # 1. But we use some checks before we call the kind class with the initializing value.
    # 2. Since we have not to memorize the value in the kind class and, instead, we
    #    converts the value in a type of the same kind class, we don't need call the kind
    #    initialization with the value.

    def __call__(cls, *args, **kwargs): # Don't write pragma @classmethod before!
        if not args and not kwargs:
            return cls
        elif args:
            value = args[0]
        else:
            value = kwargs['colon']
        if isinstance(value, cls):
            return value
        if DO_TYPECHECK:
            if value == ":":
                return super().__call__() #  sind 2. we must not call super().__call__(value)
            raise ValueError("Value is not a ColonType or a string of ':'.")
        return super().__call__() # DO_TYPECHECK == False
#
# ColonType implementation
# ------------------------
#

class ColonType(metaclass=_ColonType_Meta):
    """Type for a string what allows only a colon."""

    def __new__(cls):
        return cls
    
    def __repr__(self):
        return ":"
    
    def __str__(self):
        return ":"

#
# Ending: ColonType
#
# ############################################################


# ############################################################
#
# SType
# =====
#

#
# We need a helper "Meta" class in order to make isinstance(obj, SType)
# more duck_typing.
# Metaclass: _SType_Meta
# ----------------------
#

class _SType_Meta(type):
    @classmethod
    def __instancecheck__(cls, obj):
        # We check for the restricted format version of the shape description
        #      >>   (tuple[int >= 0, ColonType, EllipsisType])   <<
        # Thats the format after a _to_stype() call in SType.
        #
        if DO_TYPECHECK:
            if isinstance(obj, tuple):
                if len(obj) > 0:        # tuple has elements
                    for element in obj:
                        if isinstance(element, int):
                            if element >= 0:
                                continue    # element is non-negativ integer
                        elif isinstance(element, ColonType):
                            continue
                        # elif element == ":":
                        #     continue    # the last possible
                        elif isinstance(element, EllipsisType):
                            continue    # element ok: Ellipsis
                        else:
                            False       # element is not valid :-(
                else:
                    return False        # tuple is empty :-(
            else:
                return False            # no tuple :-(
            return True                 # Done: Ok. :-)
        return True # DO_TYPECHECK == False
    
    def __call__(cls, *args, **kwargs): # Don't write pragma @classmethod before!
        if not args and not kwargs:
            return cls
        elif args:
            value = args[0]
        else:
            value = kwargs['any_shape_type']
        return super().__call__(value)

#
# SType implementation
# --------------------
#

class SType(metaclass=_SType_Meta):
    """Shape format descriptor as type with conversion from less restrictec formats."""

    def __init__(self, any_shape_type: Any):
        if not isinstance(any_shape_type, SType):
            self._stype = self._to_stype(any_shape_type)
        else:
            self._stype = any_shape_type

    @staticmethod
    def _to_stype(shape: Any) -> "SType":
        """Make any object to SType as a more standadised writing of the strictly typed shape."""
        # Note: Since we change a positive non-integer into signed intager by a loop and indexing
        #       at the end, we create a list at first and convert it to a tuple at last. (Tuple
        #       elements can not be overwritten.)
        VALUE_ERROR_TXT = "Not a valid shape."

        if DO_TYPECHECK:
            if isinstance(shape, str):
                try:
                    # we have a string with or without a list inside; have to convert
                    shape = re.sub(
                        r"[\[\(\)\] ]", "", shape
                    )  # remove list/tuple brackets ans spaces
                    if (
                        len(re.sub("[0-9:.,]", "", shape)) > 0
                    ):  # mask signs which should not be in the string
                        raise ValueError(VALUE_ERROR_TXT)
                    shape = shape.split(",")
                    new_shape = []
                    for element in shape:
                        if element == "":
                            continue
                        elif element == ":":
                            new_shape.append(":")
                            continue
                        elif element == "...":
                            new_shape.append(...)
                            continue
                        # now we make it safe to have a really index integer and not a floating point value
                        # or negative values
                        if float(element) != abs(int(element)):
                            raise ValueError(VALUE_ERROR_TXT)
                        new_shape.append(int(element))
                    shape = new_shape
                except:
                    raise ValueError(VALUE_ERROR_TXT)
            # we have no string, so we could have: int | str | list[str|int|EllipsisType] | tuple[str|int|EllipsisType]
            elif not isinstance(shape, list | tuple):
                # this should be a single value:
                #   - unsigned integer
                #   - floating point, interpretable as unsigned integer
                #   - ellipsis
                try:
                    if isinstance(shape, EllipsisType):
                        shape = [...]
                    elif int(shape) >= 0:
                        shape = [int(shape)]
                    else:
                        raise ValueError(VALUE_ERROR_TXT)
                except:
                    ValueError(VALUE_ERROR_TXT)
            elif isinstance(shape, tuple):
                try:
                    shape = list(shape)
                except:
                    ValueError(VALUE_ERROR_TXT)
            # we should have a tuple now; but maybe with wrong content
            ellipsis_cnt = 0
            try:
                for i in range(len(shape)):
                    if isinstance(shape[i], EllipsisType):
                        if (i == 0) or (i == len(shape) - 1):
                            ellipsis_cnt += 1
                            continue
                        else:
                            raise ValueError(VALUE_ERROR_TXT)
                    elif shape[i] == ":":
                        continue
                    # now we make it safe to have a really index integer and not a floating point value
                    # or negative values
                    elif isinstance(shape[i], int):
                        if shape[i] < 0:
                            raise ValueError(VALUE_ERROR_TXT)
                        shape[i] = int(shape[i])
                        continue
                    else:
                        raise ValueError(VALUE_ERROR_TXT)
                if ellipsis_cnt > 1:
                    raise ValueError(VALUE_ERROR_TXT)
            except:
                raise ValueError(VALUE_ERROR_TXT)
            return tuple(shape)
        else: 
            # DO_TYPECHECK == False
            # -> We do the same but without checks.
            # This code part is copied from above after testing and shortened
            # but hast to test with __debug__ == False (optimize flag of Python)
            if isinstance(shape, str):
                # we have a string with or without a list inside; have to convert
                shape = re.sub(
                    r"[\[\(\)\] ]", "", shape
                )  # remove list/tuple brackets ans spaces
                shape = shape.split(",")
                new_shape = []
                for element in shape:
                    if element == "":
                        continue
                    elif element == ":":
                        new_shape.append(":")
                    elif element == "...":
                        new_shape.append(...)
                    else:
                        new_shape.append(int(element))
                shape = new_shape
            elif not isinstance(shape, list | tuple):
                # this should be a single value:
                #   - unsigned integer
                #   - floating point, interpretable as unsigned integer
                #   - ellipsis
                if isinstance(shape, EllipsisType):
                    shape = [...]
                else:
                    shape = [int(shape)]
            else:
                shape = list(shape)
            for i in range(len(shape)):
                if isinstance(shape[i], EllipsisType):
                    continue
                elif shape[i] == ":":
                    continue
                shape[i] = int(shape[i])
                continue
            return tuple(shape)
            
    def __repr__(self):
        return repr(self._stype)
    
    def __eq__(self, other):
        if isinstance(other, tuple):
            return self._stype == other
        if isinstance(other, SType):
            return self._stype == other._stype
        return False
    
    def __iter__(self):
        return iter(self._stype)
    
    def __getitem__(self, index):
        return self._stype[index]
    
    def __len__(self):
        return len(self._stype)
    
    def __str__(self):
        return str(self._stype)
    
    def __contains__(self, item):
        return item in self._stype
    
    def check_ndarray(self, array: ArrayLike) -> bool:
        if isinstance(array, np.ndarray):
            a_shape = array.shape
        else:
            a_shape = np.array(array).shape
        stype = self._stype # maybe we cut out ellipsis
        try:
            # we remove dimensions in case an ellipsis is given at the
            # beginning or the end
            if isinstance(stype[0], EllipsisType):
                # remove outer dimensions
                stype = stype[1:]
                a_shape = a_shape[(len(a_shape)-len(stype)):]
            elif isinstance(stype[-1], EllipsisType):                                         
                # remove inner dimensions
                stype = stype[:-1]
                a_shape = a_shape[: len(stype)]
            # now the number of dimensions should be identically
            if len(a_shape) != len(stype):
                return False
            # and finally, we check the shape step by step
            for s, a_s in zip(stype, a_shape, strict=False):
                if s == ":":
                    # Ok. It can be any size at this dimension.
                    continue
                elif s == a_s:
                    # Ok. Same size of this dimension.
                    continue
                else:
                    return False
        except:
            # We do not catch all special cases individually that
            # lead to false.
            return False

        # if we are here, so shape is ok
        return True

#
# Ending: SType
#
# ############################################################

