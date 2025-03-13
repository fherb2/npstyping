from npstyping.npstyping import (
    _Colon_Meta,
    Colon,
    _STypeLike_Meta,
    STypeLike,
    _SType_Meta,
    SType,
    sndarray,
)
import numpy as np

arr = np.arange(5)
obj = sndarray(arr, stype=(":"), auto_shape_check = False)
print(type(obj))
print(obj.auto_shape_check)
print(obj.stype)
v = obj[1:]
print(type(v))
print(v.stype)
