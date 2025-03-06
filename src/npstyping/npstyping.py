import re
from types import EllipsisType
from typing import Literal
import numpy as np
from numpy.typing import ArrayLike


# ############################################################
#
# Common / simple type definitions, Constants
# ===============================
#

# TODO: Look if we have to use it (or asserts instead)
DO_TYPECHECK = False
if __debug__:
    DO_TYPECHECK = True

NPOrder = Literal["C", "F", "A", "K"]


class ShapeError(Exception):
    pass


#
# Ending: Common / simple type definitions, Constants
#
# ############################################################

# ############################################################
#
# Colon (Type)
# ============
#

#
# We use a helper "Meta" class in order to make isinstance(obj, Colon)
# more duck_typing: Can be a Colon object or a ":" string.
# Metaclass: _Colon_Meta
# ----------------------
#

class _Colon_Meta(type):
    @classmethod
    def __instancecheck__(cls, obj):
        # We check for the content, a single valid sign:
        #       >>   ":"   <<
        if isinstance(obj, cls):
            return True
        elif obj == ":":
            return True
        return False

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

    def __call__(cls, *args, **kwargs):  # Don't write pragma @classmethod before!
        if not args and not kwargs:
            return cls
        elif args:
            value = args[0]
        else:
            value = kwargs["colon"]
        if isinstance(value, cls):
            return value
        if value == ":":
            return (
                super().__call__()
            )  #  since 2, we must not call super().__call__(value)
        raise ValueError("Value is not a Colon (type class) or a string of ':'.")

#
# Colon implementation
# --------------------
#

class Colon(metaclass=_Colon_Meta):
    """Colon using in SType.
    
    Class can be used to check for a colon independing the colon
    from this class or Literal[":"] by using isinstance(). 
    """

    def __new__(cls):
        return cls

    def __repr__(self):
        return ":"

    def __str__(self):
        return ":"

#
# Ending: Colon (Type)
#
# ############################################################

# ############################################################
#
# STypeLike
# =========
#
# It is STypeLike if
#   - it is SType or STypeLike object
#   - can be converted to SType
#
# To be fast in checking, we check in following order:
#   - isinstance(..., SType)
#   - is formatted as SType (tuple of...)
#   - is convertible into SType (Any...)
#


class _STypeLike_Meta(type):
    @staticmethod
    def _filter_brackets_spaces_from_string(obj:str) -> str:
        obj = re.sub("[\s]", "", obj)
        m1 = re.match(r"^[\[](.*)[\]]$", obj)
        m2 = re.match(r"^[\()](.*)[\)]$", obj)
        m3 = re.match(r"^[\{](.*)[\}]$", obj)
        for m in [m1, m2, m3]:
            if m:
                obj = m.group(1)
                break
        return obj

    @classmethod
    def __instancecheck__(cls, obj):
        try:
            # Is instance of SType (includes None, bool, tuple and SType
            # class, since its tested in _SType_Meta?
            if isinstance(obj, SType | cls):
                return True

            # is convertible into SType?
            if isinstance(obj, str):
                # we have a string with or without a list inside; have to convert
                obj = cls._filter_brackets_spaces_from_string(obj)
                if (
                    len(obj) == 0 or len(re.sub("[0-9:.,]", "", obj)) > 0
                ):  # mask signs which should not be in the string
                    return False
                obj = obj.split(",")
                for element in obj:
                    if element == "":
                        continue
                    elif element == ":":
                        continue
                    elif element == "...":
                        continue
                    # now we make it safe to have a really positiv integer and not a floating point value
                    # or negative values
                    if float(element) != abs(int(element)):
                        return False
            # we have no string, so we could have: int | str | list[str|int|EllipsisType] | tuple[str|int|EllipsisType]
            elif not isinstance(obj, list | tuple):
                # this should be a single value:
                #   - unsigned integer
                #   - floating point, interpretable as unsigned integer
                #   - ellipsis
                #   - Colon
                if not isinstance(obj, (EllipsisType, Colon)) and (int(obj) < 0):
                    return False
            else:
                # a list or tuple

                # content of each element has to be
                #   - unsigned integer,
                #   - floating point, interpretable as unsigned integer,
                #   - ellipsis or
                #   - Colon
                for element in obj:
                    if not isinstance(element, (EllipsisType, Colon)) and (
                        int(element) < 0
                    ):
                        return False
                    else:
                        continue
        except Exception:
            return False

        return True


#
# SType_like implementation
# --------------------------
#


class STypeLike(metaclass=_STypeLike_Meta):
    """Class which is SType or can be converted into SType."""

    def __new__(cls):
        return cls

    def __repr__(self):
        return self._stype_like

    def __str__(self):
        return str(self._stype_like)


#
# Ending: STypeLike
#
# ############################################################


# ############################################################
#
# SType
# =====
#
# In addition to the description of the shape itself, SType can also
# describe special cases as sndarray parameter.
#
# The proper meaning is:
#
#   (":", 1) or other specifications – normal use case as: Shape restriction.
#
# Additional meaning in context with sndarray use:
#
#   None – No restriction. Don't use the actual shape as new SType
#          restriction. Wait until the shape constraint is
#          set at a later point. Shape checks will be return True.
#   True – Take the actual shape of the array and use it as constraint.
#          This is a trigger: If set this to True, the value will
#          be replaced by the actual array shape.
#   False – Reserved for future use. No meaning
#
# Values
# ------
#
#   - stype – Value in the right format and usable as shape type in sndarray
#             and to check a numpy.ndarray directly for a special shape.
#             The "right format" is a tuple of elements of type
#               - integer with value >= 0
#               - EllipsisType '...'
#               - Colon (":").
#             And Ellipsis may be only the first or/and the last element in
#             tuple.
#
# Methods
# -------
#
#   -   SType   – (defined as: __new__() ); get a SType class
#
#   -   SType(value: STypeLike) (defines as __init__() );  returns a SType class instance
#                 with initialized value 'stype'
#
#   -   check_ndarray() – check a NumPy (like) array against the shape type 'stype'
#
# Since value 'stype' is a tuple, we handle it as a tuple also:
#   -   __iter__()
#   -   __getitem__()
#   -   __len__()
#   -   __contains__()
#
#   -   __str__()   – value 'stype' as string
#
#   -   __repr__()  – representation of the value 'stype' as tuple
#
#   -   __eq__()    – equal compare function; is equal if it is equal with the
#                     stype attribute of a SType object or a tuple with the
#                     correct content in the sense of SType (duck-typing manner)
#
#   -   _to_stype() – internal helper: converts an STypeLike object into SType

#
# We use a helper "Meta" class in order to make isinstance(obj, SType)
# more duck_typing.
#
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
        if isinstance(obj, cls | None | bool):
            # Special cases. It's ok.
            return True
        if isinstance(obj, tuple):
            if len(obj) > 0:  # tuple has elements
                for element in obj:
                    if isinstance(element, int):
                        if element >= 0:
                            continue  # element is non-negativ integer
                    elif isinstance(element, Colon):
                        continue
                    # elif element == ":":
                    #     continue    # the last possible
                    elif isinstance(element, EllipsisType):
                        continue  # element ok: Ellipsis
                    else:
                        return False  # element is not valid :-(
            else:
                return False  # tuple is empty :-(
        else:
            return False  # no tuple :-(
        return True  # Done: Ok. :-)

    def __call__(cls, *args, **kwargs):  # Don't write pragma @classmethod before!
        if not args and not kwargs:
            return cls
        elif args:
            value = args[0]
        else:
            value = kwargs["any_shape_type"]
        return super().__call__(value)


#
# SType implementation
# --------------------
#


class SType(metaclass=_SType_Meta):
    """Shape format descriptor.
     
    Has the functions as:

    - tuple based data type to describe shape restrictions (as more restricted format than STypeLike)
    - converter from STypeLike shape restriction to tuple based SType
    - checker for numpy.ndarray shape against the SType shape restriction

    SType(":, 3, ...") == (Colon, 3, ...)
    SType(":, 2").check_ndarray(np.asarray([[1, 2], [3,4], [5,6]])) == True

    """

    def __init__(self, stype_like: STypeLike):
        if not isinstance(stype_like, SType):
            self._stype = self._to_stype(stype_like)
        else:
            self._stype = stype_like

    @property
    def stype(self):
        return self._stype

    @stype.getter
    def stype(self, stype_like: STypeLike):
        self._stype = self._to_stype(stype_like)

    @staticmethod
    def _to_stype(shape: STypeLike) -> "SType":
        """Make any object to SType as a more standardised writing of the strictly typed shape."""
        # Note: Since we change a positive non-integer into signed integer by a loop and indexing
        #       at the end, we create a list at first and convert it to a tuple at last. (Tuple
        #       elements can not be overwritten.)
        VALUE_ERROR_TXT = "Not a valid shape."

        if isinstance(shape, None | bool):
            return shape
        elif isinstance(shape, str):
            try:
                # we have a string with or without a list inside; have to convert
                shape = STypeLike._filter_brackets_spaces_from_string(shape)
                if (
                    len(shape) == 0 or len(re.sub("[0-9:.,]", "", shape)) > 0
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
            except Exception:
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
            except Exception:
                ValueError(VALUE_ERROR_TXT)
        elif isinstance(shape, tuple):
            try:
                shape = list(shape)
            except Exception:
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
        except Exception:
            raise ValueError(VALUE_ERROR_TXT)
        return tuple(shape)

    def check_ndarray(self, array: ArrayLike) -> bool:
        if isinstance(array, np.ndarray):
            a_shape = array.shape
        else:
            a_shape = np.array(array).shape
        stype = self._stype  # maybe we cut out ellipsis
        try:
            # we remove dimensions in case an ellipsis is given at the
            # beginning or the end
            if isinstance(stype[0], EllipsisType):
                # remove outer dimensions
                stype = stype[1:]
                a_shape = a_shape[(len(a_shape) - len(stype)) :]
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
        except Exception:
            # We do not catch all special cases individually that
            # lead to false.
            return False

        # if we are here, so shape is ok
        return True

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


#
# Ending: SType
#
# ############################################################

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
#   -   Mechanism to keep the stype attribute in such cases where
#       a numpy operation creates a new ndarray.
#
#   -   check_stype – Method to check the restriction of the shape
#                     against the current array shape


class sndarray(np.ndarray):

    def __new__(
        cls,
        a: ArrayLike,
        dtype: np.dtype | None = None,
        order: NPOrder | None = None,
        *,
        stype: STypeLike | None = None,
        auto_shape_check: bool = False,
        device: Literal['cpu'] | None = None,
        copy: bool | None = None,
        like: ArrayLike | None = None,
    ):
        # Create the numpy array
        obj = np.asarray(a, dtype, order, device=device, copy=copy, like=like).view(cls)
        # Add additional properties
        assert isinstance(stype, STypeLike), ValueError(
            "Argument 'stype' has wrong data type (not a STypeLike)."
        )
        assert isinstance(auto_shape_check, bool), ValueError(
            "Argument 'auto_shape_check' has wrong data type (not a boolean)."
        )
        obj._auto_shape_check = auto_shape_check
        obj._stype = SType(stype)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return None
        self._stype = getattr(obj, "_stype", None)
        self._auto_shape_check = getattr(obj, "_auto_shape_check", None)

    @property
    def stype(self):
        return self._stype

    @stype.setter
    def stype(self, stype_like: STypeLike | bool | None):
        if stype_like is None:
            self._stype = None
            return
        if isinstance(stype_like, bool):
            if stype_like:
                # its boolean 'True'; means: Take array's current shape as stype.
                self._stype = self.shape
            # 'False' has no meaning.
            return
        assert isinstance(stype_like, STypeLike), ValueError(
            "'stype_like' has wrong data type."
        )
        self._stype = SType(stype_like)

    def check_stype(self, stype_like: STypeLike | None = None) -> bool:
        if stype_like is not None:
            assert isinstance(stype_like, STypeLike), ValueError(
                "Parameter 'stype_like' is not STypeLike."
            )
            self._stype = SType(stype_like)
        return self._stype.check_ndarray(self.__array__(copy=False))

    #
    # parts of code to implement 'auto-check' and 'keep stype' behaviour
    # ------------------------------------------------------------------
    #

    # implement the process into all of numpy operations
    def __getattribute__(self, name):
        attr = super().__getattribute__(name)

        if callable(attr):

            def wrapper_method(*args, **kwargs):
                if self._auto_shape_check:
                    current_shape = self.shape

                result = attr(*args, **kwargs)

                if isinstance(result, np.ndarray):
                    if hasattr(result, 'device'):
                        result = sndarray( a=result, dtype=result.dtype, stype=self._stype, 
                                           auto_shape_check=self._auto_shape_check, 
                                           device=self.device)
                    else:
                        result = sndarray( a = result, dtype=result.dtype, stype=self._stype, 
                                        auto_shape_check=self._auto_shape_check)
                    
                if self._auto_shape_check \
                   and hasattr(result, 'shape') \
                   and current_shape != result.shape:
                    result._stype.check_ndarray(result.__array__(copy=False))

                return result

            return wrapper_method
        return attr

#
# Ending: sndarray (shape typed numpy.ndarray)
#
# ############################################################
