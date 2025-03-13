"""npstyping – Numpy shape typing."""  # noqa: RUF002

import re
from collections.abc import Iterator
from types import EllipsisType
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

# ############################################################
#
# Common / simple type definitions, Constants
# ===============================
#

DO_TYPECHECK = __debug__
"""If False: Switches off type- and instance-checks.

    'DO_TYPECHECK' is set to 'True' automatically, if Python is not started in
    optimizes mode (option '-O'): DO_TYPECHECK is a simple copy of __debug__.

    With 'DO_TYPECHECK' == 'False' in optimized mode, the 'isinstance'-typechecking of
    classes

      - Colon,
      - STypeLike and
      - Stype

    are disabled to safe processing time (simmilar to the removed asserts in this mode).

    If you want to use isinstance-checks also for these classes in this mode, so set

        'npstyping.DO_TYPECHECK = True'

    explicitly after importing the modul.

    Class sndarray:

    The type checks of the ndarray subclass of sndarray are unaffected from this behavior.
    Only the stype-Parameter keeps unchecked if you set it. Are wrong values used during
    this mode where type checking is switched of, so exceptions can happen.

"""

NPOrder = Literal["C", "F", "A", "K"]
"""Data type for the 'order' parameter of numpy.asarray()."""


class ShapeError(Exception):
    """An automatically strict shape type check by SType was not successfully."""


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


class _Colon_Meta(type):  # noqa: N801
    """Meta class for type class 'Colon'."""

    # Do not write '@classmethod' here!
    def __instancecheck__(cls, obj: object) -> bool:
        """Check if object is class Colon or is string with value ':'."""
        # We check for the content, a single valid sign:
        #       >>   ":"   <<
        if not DO_TYPECHECK:
            return True
        if type(obj) is Colon:
            return True
        if isinstance(obj, str) and obj == ":":
            return True
        return False

    # # The most simple form of __call__(cls, *args, **kwargs) looks like:
    # #
    # #   if not args and not kwargs:
    # #       return cls
    # #   return super().__call__(*args, **kwargs)
    # #
    # # 1. But we use some checks before we call the kind class with the initializing value.
    # # 2. Since we have not to memorize the value in the kind class and, instead, we
    # #    converts the value in a type of the same kind class, we don't need call the kind
    # #    initialization with the value.

    # def __call__(cls, *args, **kwargs):  # Don't write pragma @classmethod before!
    #     if not args and not kwargs:
    #         return cls
    #     elif args:
    #         value = args[0]
    #     else:
    #         value = kwargs["colon"]
    #     if isinstance(value, cls):
    #         return value
    #     if value == ":":
    #         return (
    #             super().__call__()
    #         )  #  since 2, we must not call super().__call__(value)
    #     raise ValueError("Value is not a Colon (type class) or a string of ':'.")


#
# Colon implementation
# --------------------
#


class Colon(metaclass=_Colon_Meta):
    """Colon using in SType.

    Class can be used to check for a colon independing the colon
    from this class or Literal[":"] by using isinstance().
    """

    def __repr__(self) -> str:
        """Return Colon type (is represented by character ':')."""
        return "The Colon (as type or as string ':') is used to specify a dimension size as any value. But the dimension must be present."

    def __str__(self) -> str:
        """Return string representation of Colon type is ':'."""
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


def _filter_brackets_spaces_from_string(obj: str) -> str:
    """Remove pairs of brackets in strings.

    Helper function only.

    """
    obj = re.sub(r"[\s]", "", obj)
    m1 = re.match(r"^[\[](.*)[\]]$", obj)
    m2 = re.match(r"^[\()](.*)[\)]$", obj)
    m3 = re.match(r"^[\{](.*)[\}]$", obj)
    for m in [m1, m2, m3]:
        if m:
            obj = m.group(1)
            break
    return obj


class _STypeLike_Meta(type):  # noqa: N801
    """Meta class for type class 'STypeLike'."""

    # Do not write '@classmethod' here!
    def __instancecheck__(cls, obj: object) -> bool:  # noqa: C901
        """Check if object is class STypeLike or a value what is convertible to SType."""
        if not DO_TYPECHECK:
            return True
        try:
            # Is instance of SType (includes None, bool, tuple and SType
            # class, since its tested in _SType_Meta?
            if type(obj) is STypeLike:
                return True

            # is convertible into SType?
            if isinstance(obj, str):
                # we have a string with or without a list inside; have to convert
                obj = _filter_brackets_spaces_from_string(obj)
                if (
                    len(obj) == 0 or len(re.sub("[0-9:.,]", "", obj)) > 0
                ):  # mask signs which should not be in the string
                    return False
                obj = obj.split(",")
                for element in obj:
                    if element in ("", ":", "..."):
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
                if not isinstance(obj, EllipsisType | Colon) and (int(obj) < 0):
                    return False
            else:
                # a list or tuple

                # content of each element has to be
                #   - unsigned integer,
                #   - floating point, interpretable as unsigned integer,
                #   - ellipsis or
                #   - Colon
                for element in obj:
                    if not isinstance(element, EllipsisType | Colon) and (
                        int(element) < 0
                    ):
                        return False
                    continue
        except Exception:  # noqa: BLE001
            return False

        return True


#
# SType_like implementation
# --------------------------
#


class STypeLike(metaclass=_STypeLike_Meta):
    """Type class for values which are convertible to SType.

    The only function is typechecking of an object via "isinstance".
    It returns True in case the value is of type SType or convertible
    into SType (by using class SType).

    """


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


class _SType_Meta(type):  # noqa: N801
    """Meta class for type class 'SType'."""

    # Do not write '@classmethod' here!
    def __instancecheck__(cls, obj: object) -> bool:  # noqa: C901
        """Check if object is of type SType or has the correct signature as tuple.

        The generalised signature is: tuple[int >= 0, Colon, EllipsisType]

        But type can also be a boolean or None for special cases by using in sndarray.

        This signature is the format/type after convertion with _to_stype() what will
        be used by initializing or setting the value of a SType class instance.

        """
        # We check for the restricted format version of the shape description
        #      >>   (tuple[int >= 0, ColonType, EllipsisType])   <<
        # Thats the format after a _to_stype() call in SType.
        #
        if not DO_TYPECHECK:
            return True
        if type(obj) is SType:
            return True
        if isinstance(obj, None | bool):
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
                    elif isinstance(element, EllipsisType):
                        continue  # element ok: Ellipsis
                    else:
                        return False  # element is not valid :-(
            else:
                return False  # tuple is empty :-(
        else:
            return False  # no tuple :-(
        return True  # Done: Ok. :-)

    def __call__(
        cls,
        *args,
        **kwargs,
    ) -> "_SType_Meta":  # Don't write pragma @classmethod before this method!
        if not args and not kwargs:
            return cls
        value = args[0] if args else kwargs["stype_like"]
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

    def __init__(self, stype_like: STypeLike) -> None:
        """Create a instance of SType with a value what is of type SType or convertible into it."""
        if not isinstance(stype_like, SType):
            self._stype = self._to_stype(stype_like)
        else:
            self._stype = stype_like

    @property
    def stype(self) -> tuple[Colon | int | EllipsisType]:
        """Get the SType value."""
        return self._stype

    @stype.getter
    def stype(self, stype_like: STypeLike) -> None:
        """Set SType with a SType value or a convertible value (STypeLike)."""
        if isinstance(stype_like, SType):
            self._stype = stype_like
        else:
            self._stype = self._to_stype(stype_like)

    @classmethod
    def _to_stype(cls, shape: STypeLike) -> "SType":  # noqa: C901
        """Make any STypeLike object to a SType signature as a more standardised writing of the strictly typed shape."""
        # Note: Since we change a positive non-integer into signed integer by a loop and indexing
        #       at the end, we create a list at first and convert it to a tuple at last. (Tuple
        #       elements can not be overwritten.)
        try:
            if isinstance(shape, cls | None | bool):
                return shape
            if isinstance(shape, str):
                # we have a string with or without a list inside; have to convert
                shape = _filter_brackets_spaces_from_string(shape)
                if (
                    len(shape) == 0 or len(re.sub("[0-9:.,]", "", shape)) > 0
                ):  # mask signs which should not be in the string
                    raise Exception  # noqa: TRY002, TRY301
                shape = shape.split(",")
                new_shape = []
                for element in shape:
                    if element == "":
                        continue
                    if element == ":":
                        new_shape.append(":")
                        continue
                    if element == "...":
                        new_shape.append(...)
                        continue
                    # now we make it safe to have a really index integer and not a floating point value
                    # or negative values
                    if float(element) != abs(int(element)):
                        raise Exception  # noqa: TRY002, TRY301
                    new_shape.append(int(element))
                shape = new_shape
            # we have no string, so we could have: int | str | list[str|int|EllipsisType] | tuple[str|int|EllipsisType]
            elif not isinstance(shape, list | tuple):
                # this should be a single value:
                #   - unsigned integer
                #   - floating point, interpretable as unsigned integer
                #   - ellipsis

                if isinstance(shape, EllipsisType):
                    shape = [...]
                elif int(shape) >= 0:
                    shape = [int(shape)]
                else:
                    raise Exception  # noqa: TRY002, TRY301
            elif isinstance(shape, tuple):
                shape = list(shape)
            # we should have a tuple now; but maybe with wrong content
            ellipsis_cnt = 0
            for i in range(len(shape)):
                if isinstance(shape[i], EllipsisType):
                    if (i == 0) or (i == len(shape) - 1):
                        ellipsis_cnt += 1
                        continue
                    raise Exception  # noqa: TRY002, TRY301
                if shape[i] == ":":
                    continue
                # now we make it safe to have a really index integer and not a floating point value
                # or negative values
                if isinstance(shape[i], int):
                    if shape[i] < 0:
                        raise Exception  # noqa: TRY002, TRY301
                    shape[i] = int(shape[i])
                    continue
                raise Exception  # noqa: TRY002, TRY301
            if ellipsis_cnt > 1:
                raise Exception  # noqa: TRY002, TRY301
            return tuple(shape)
        except Exception:  # noqa: BLE001
            msg = "Not a valid shape."
            raise ValueError(msg)  # noqa: B904

    def check_ndarray(self, array: ArrayLike) -> bool:
        """Check an numpy array(-like) object for the shape.

        The return value is a boolean. No Exception will be raised if
        the chape is not correct.

        """
        if isinstance(array, np.ndarray):
            a_shape = array.shape
        else:
            a_shape = np.array(array).shape
        stype = self._stype  # maybe we cut out ellipsis later, so we copy it here
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
                if s == a_s:
                    # Ok. Same size of this dimension.
                    continue
                return False
        except Exception:  # noqa: BLE001
            # We do not catch all special cases individually that
            # lead to false.
            return False

        # if we are here, so shape is ok
        return True

    def __eq__(self, other: STypeLike | bool | None) -> bool:
        """Check the value against an other valid value."""
        if isinstance(other, tuple | bool | None):
            return self._stype == other
        if isinstance(other, SType):
            return self._stype == other._stype
        if isinstance(other, STypeLike):
            return self._stype == SType(other)
        return False

    def __iter__(self) -> Iterator:
        """Implement iter(self)."""
        return iter(self._stype)

    def __getitem__(self, index: int) -> Colon | EllipsisType | int:
        """Implement getitem(self)."""
        return self._stype[index]

    def __len__(self) -> int:
        """Return len(self)."""
        return len(self._stype)

    def __str__(self) -> str:
        """Return the string representation."""
        return str(self._stype)

    def __contains__(self, item: Any) -> bool:  # noqa: ANN401
        """Return key in self."""
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


class sndarray(np.ndarray):  # noqa: N801, Compatible naming to type numpy.ndarray
    """Numpy array with shape restiction behavior."""

    # implementation see: https://numpy.org/doc/2.1/user/basics.subclassing.html
    def __new__(
        cls,
        a: ArrayLike,
        dtype: np.dtype | None = None,
        order: NPOrder | None = None,
        *,
        stype: STypeLike | None = None,
        auto_shape_check: bool = False,
        device: Literal["cpu"] | None = None,
        copy: bool | None = None,
        like: ArrayLike | None = None,
    ) -> "sndarray":
        """Create sndarray class."""
        # Check inputs
        assert isinstance(stype, STypeLike), ValueError(
            "Argument 'stype' has wrong data type (not a STypeLike).",
        )
        assert isinstance(auto_shape_check, bool), ValueError(
            "Argument 'auto_shape_check' has wrong data type (not a boolean).",
        )
        # Create the numpy array
        obj = np.asarray(a, dtype, order, device=device, copy=copy, like=like).view(cls)
        # Add additional properties
        obj.auto_shape_check = auto_shape_check
        obj.stype = SType(stype)
        return obj

    def __array_finalize__(self, obj: object) -> None:
        """Finalize the array."""
        if obj is None:
            return
        self.stype = getattr(obj, "stype", None)
        self.auto_shape_check = getattr(obj, "auto_shape_check", None)

    @property
    def stype(self) -> SType:
        """Return stype attribute."""
        return self._stype

    @stype.setter
    def stype(self, stype_like: STypeLike | bool | None) -> None:
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
            "'stype_like' has wrong data type.",
        )
        self._stype = SType(stype_like)

    def check_stype(self, stype_like: STypeLike | None = None) -> bool:
        """Check the array by shape restrictions.

        Returns the result, but doesn't raise a ShapeError exception.

        Parameters
        ----------
        stype_like : STypeLike | None, optional
            Set the 'stype' property before the check. Otherwise uses the
            value of this property, by default None

        Returns
        -------
        bool
            True, if the arrays shape is positive validated by the shape
            type property 'stype'. Otherwise false.

        """
        if stype_like is not None:
            assert isinstance(stype_like, STypeLike), ValueError(
                "Parameter 'stype_like' is not STypeLike.",
            )
            self._stype = SType(stype_like)
        return self._stype.check_ndarray(self.__array__(copy=False))

    #
    # parts of code to implement 'auto-check' and 'keep stype' behaviour
    # ------------------------------------------------------------------
    #

    # implement the process into all of numpy operations
    def __getattribute__(self, name: Any) -> Any:  # noqa: ANN401
        """Overwrtes the method to add some functionality around the numpy methods."""
        attr = super().__getattribute__(name)

        if callable(attr):

            def wrapper_method(*args, **kwargs):  # noqa: ANN202
                """Add some functionality around the numpy methods."""
                if self._auto_shape_check:
                    current_shape = self.shape

                result = attr(*args, **kwargs)

                if isinstance(result, np.ndarray):
                    if hasattr(result, "device"):
                        result = sndarray(
                            a=result,
                            dtype=result.dtype,
                            stype=self._stype,
                            auto_shape_check=self._auto_shape_check,
                            device=self.device,
                        )
                    else:
                        result = sndarray(
                            a=result,
                            dtype=result.dtype,
                            stype=self._stype,
                            auto_shape_check=self._auto_shape_check,
                        )

                if (
                    self._auto_shape_check
                    and hasattr(result, "shape")
                    and current_shape != result.shape
                ):
                    result.stype.check_ndarray(result.__array__(copy=False))

                return result

            return wrapper_method
        return attr


#
# Ending: sndarray (shape typed numpy.ndarray)
#
# ############################################################
