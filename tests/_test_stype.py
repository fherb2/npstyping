

import numpy as np
import pytest
from srcnpstyping.npstyping import Colon, STypeLike, SType, sndarray



# ############################################
#
# Typed-Meta implementations
# --------------------------
#

# This class works as template. We test it with the
# implementation of class 'ColonType'
def test__ColonType_Meta():
    pass

# This class works as template. We test it with the
# implementation of class 'SType'
def test__SType_Meta():
    pass

#
# Ending: Typed-Meta implementations
#
# ############################################

# ############################################
#
# ColonType
# ---------
#

def test_ColonType_types():
    assert type(Colon) == _ColonType_Meta
    a = Colon
    b = Colon(":")
    assert isinstance(a, Colon)
    assert isinstance(b, Colon)
    
@pytest.mark.parametrize("in1, out1", [  
        (":",           True),
        (Literal[":"],  True),
        ("",            False),
        ("1",           False),
        (1,             False),
    ]) 
def test_ColonType(in1, out1):
    assert isinstance(in1, Colon) == out1

@pytest.mark.parametrize("in1", [  
        (";"),
        (1)               
    ])
def test_ColonType_value_error_exceptions(in1):
    with pytest.raises(ValueError):
        Colon(in1)


#
# Ending: ColonType
#
# ############################################


# ############################################
#
# SType
# -----
#

def test_SType_type():
    assert type(SType) == _SType_Meta
    a = SType
    b = SType("2")
    isinstance(a, SType)
    isinstance(b, SType)

positiv_test_list = [
        ([3], (3,)),
        ("[3]", (3,)),
        ("(3)", (3,)),
        ("3", (3,)),
        ([":"], (":",)),
        ("[:]", (":",)),
        ("(:)", (":",)),
        (":", (":",)),
        ([...], (...,)),
        (..., (...,)),
        ("...", (...,)),
        ("[...]", (...,)),
        ("(...)", (...,)),
        ([3, 2], (3, 2)),
        ("3,2", (3, 2)),
        ("[3, 2]", (3, 2)),
        ([":", 2], (":", 2)),
        ((":", 2), (":", 2)),
        ("[:, 2]", (":", 2)),
        ("([:, 2])", (":", 2)),
        ("(:, 2)", (":", 2)),
        (":, 2", (":", 2)),
        ("(:, 2)", (":", 2)),
        ([..., 2], (..., 2)),
        ("[..., 2]", (..., 2)),
        ("..., 2", (..., 2)),
        ([3, ...], (3, ...)),
        ("[3, ...]", (3, ...)),
        ("3, ...", (3, ...)),
        ([":", ...], (":", ...)),
        ("[:, ...]", (":", ...)),
        (":, ...", (":", ...)),
        ("(:, ...)", (":", ...)),
        ([..., 3, 10], (..., 3, 10)),
        ("[..., 3, 10]", (..., 3, 10)),
        ("(..., 3, 10)", (..., 3, 10)),
        ("..., 3, 10", (..., 3, 10)),
        ([":", 3, 10], (":", 3, 10)),
        ("(:, 3, 10)", (":", 3, 10)),
        ("[:, 3, 10]", (":", 3, 10)),
        (":, 3, 10", (":", 3, 10)),
        ([..., ":", 10], (..., ":", 10)),
        ("[..., :, 10]", (..., ":", 10)),
        ("..., :, 10", (..., ":", 10)),
        ("(..., :, 10)", (..., ":", 10)),
        ([..., 10], (..., 10)),
        ("[..., 10]", (..., 10)),
        ("..., 10", (..., 10)),
        ("(..., 10)", (..., 10)),
        ([..., 3, ":"], (..., 3, ":")),
        ("[..., 3, :]", (..., 3, ":")),
        ("..., 3, :", (..., 3, ":")),
        ("(..., 3, :)", (..., 3, ":")),
    ]

@pytest.mark.parametrize("in1", positiv_test_list[:][0])
def test_SType_check_instance(in1):
    isinstance(in1, SType)

@pytest.mark.parametrize("in1, out1", positiv_test_list)   
def test_SType_check_result(in1, out1):
    assert SType(in1) == out1

array_test_list = [
        ([3], np.array((1,2,3)), True),
        ("[3]", np.array((1,1,2)), True),
        ("(3)", np.array(([1,0], [1,2], [2,2])), False),
        ("3", np.array(([1,2,3,4],)), False),
        ([":"], np.array((1,)), True),
        ("[:]", np.array((1,2,3,4,5)), True),
        ("(:)", np.array(([1],[2])), False),
        (":", np.array(([1,2],[2,3])), False),
        ([...], np.array([]), True),
        (..., np.array([1,2,3]), True),
        ("...", np.array(([1],[2],[3])), True),
        ("[...]", np.array(([1,2], [2,2], [3,3])), True),
        ("(...)", np.array(([1,2], [2,2], [3,3])), True),
        ([3, 2], np.array(([1,2], [2,2], [2,3])), True),
        ("3,2", np.array(([1,2], [2,2])), False),
        ("[3, 2]", np.array(([1], [2], [2])), False),
        ([":", 2], np.array(([1,2], [2,2])), True),
        ((":", 2), np.array(([1,2,3], [2,2,3])), False),
        ("[:, 2]", np.array(([1,2],)), True),
        ("([:, 2])", np.array([]), False),
        ("(:, 2)", np.array(([1,2], [2,2])), True),
        ([..., 2], np.array(([[1,2], [2,2]])), True),
        ("[..., 2]", np.array(([[1,2,3], [2,2,3]],)), False),
        ("..., 2", np.array((1,2)), True),
        ([3, ...], np.array((3, 2, 1)), True),
        ("[3, ...]", np.array(((3, 2, 1), (3, 2, 1), (3, 2, 1))), True),
        ("3, ...", np.array(((3,), (3,), (3,))), True),
        ([":", ...], np.array((([3, 2], [1, 4]), ([3, 2], [1, 4]), ([3, 2], [1, 4]))), True),
        ("[:, ...]", np.array([]), True),
        (":, ...", np.array([1,2,3]), True),
        ("(:, ...)", np.array([[1,2,3]]), True),
        ([..., 3, 1], np.array([[1], [2], [3]]), True),
        ("[..., 3, 1]", np.array([1]), False),
        ("(..., 3, 1)", np.array(([[1],[2],[3]], [[1],[2],[3]], [[1],[2],[3]])), True),
        ("..., 3, 1", np.array(([[1]], [[2]], [[3]])), False),
        ([":", 3, 1], np.array(([[1],[2],[3]], [[1],[2],[3]])), True),
        ("(:, 3, 1)", np.array(([[1],[2],[3]],)), True),
        ("[:, 3, 1]", np.array([[1],[2],[3]]), False),
        (":, 3, 1", np.array((([1]),)), False),
        ([..., ":", 1], np.array(([1], [2])), True),
        ("[..., :, 1]", np.array(([[1],[2],[3]], [[1],[2],[3]], [[1],[2],[3]])), True),
        ("..., :, 1", np.array(([[1,2,3], [2,2,3]])), False),
        ("(..., :, 1)", np.array(([1],[2],[3])), True),
        ([..., 1], np.array([([1],)]), True),
        ("[..., 1]", np.array([1]), True),
        ("..., 1", np.array([]), False),
        ("(..., 1)", np.array((1, 2)), False),
        ([..., 3, ":"], np.array(([[1],[2],[3]], [[1],[2],[3]], [[1],[2],[3]])), True),
        ("[..., 3, :]", np.array(([[],[],[]], [[],[],[]], [[],[],[]])), True),
        ("..., 3, :", np.array([1,2,3]), False),
        ("(..., 3, :)", np.array([[1],[2],[3]]), True)
    ]

@pytest.mark.parametrize("in1, in2, out1", array_test_list)
def test_SType_method_check_ndarray(in1, in2, out1):
    assert SType(in1).check_ndarray(in2) == out1
    
#
# Ending: SType
#
# ############################################

