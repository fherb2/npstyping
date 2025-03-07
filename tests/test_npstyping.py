import numpy as np
import pytest

from npstyping.npstyping import (
    _Colon_Meta,
    Colon,
    _STypeLike_Meta,
    STypeLike,
    _SType_Meta,
    SType,
    sndarray,
)


# ############################################
#
# ColonType
# ---------
#


def test_ColonType_types():
    assert isinstance(Colon, _Colon_Meta)
    a = Colon
    assert isinstance(a, Colon)
    b = Colon(":")
    assert isinstance(b, Colon)


@pytest.mark.parametrize(
    "in1, out1",
    [
        (":", True),
        ("", False),
        ("1", False),
        (1, False),
    ],
)
def test_ColonType(in1, out1):
    assert isinstance(in1, Colon) == out1


@pytest.mark.parametrize("in1", [(";"), (1)])
def test_ColonType_value_error_exceptions(in1):
    with pytest.raises(ValueError):
        Colon(in1)


#
# Ending: ColonType
#
# ############################################

# ############################################
#
# STypeLike
# ---------
#

# for acceptance as STypeLike values
STypeLike_positiv_test_list = [
    ([3]),
    ("[3]"),
    ("(3)"),
    ("3"),
    ([":"]),
    ("[:]"),
    ("(:)"),
    (":"),
    ([...]),
    (...),
    ("..."),
    ("[...]"),
    ("(...)"),
    ([3, 2]),
    ("3,2"),
    ("[3, 2]"),
    ([":", 2]),
    ((":", 2)),
    ("[:, 2]"),
    ("(:, 2)"),
    (":, 2"),
    ("(:, 2)"),
    ([..., 2]),
    ("[..., 2]"),
    ("..., 2"),
    ([3, ...]),
    ("[3, ...]"),
    ("3, ..."),
    ([":", ...]),
    ("[:, ...]"),
    (":, ..."),
    ("(:, ...)"),
    ([..., 3, 10]),
    ("[..., 3, 10]"),
    ("(..., 3, 10)"),
    ("..., 3, 10"),
    ([":", 3, 10]),
    ("(:, 3, 10)"),
    ("[:, 3, 10]"),
    (":, 3, 10"),
    ([..., ":", 10]),
    ("[..., :, 10]"),
    ("..., :, 10"),
    ("(..., :, 10)"),
    ([..., 10]),
    ("[..., 10]"),
    ("..., 10"),
    ("(..., 10)"),
    ([..., 3, ":"]),
    ("[..., 3, :]"),
    ("..., 3, :"),
    ("(..., 3, :)"),
    ("{..., 3, :}"),
]

STypeLike_negativ_test_list = [
    ({"a": 3}),
    ("[3"),
    ("3)"),
    ("3.1"),
    (":]"),
    ("()"),
    ("[..]"),
    ("[....]"),
    ("[..., :, -10]"),
    ("...., :, 10"),
    ("(.., :, 10)"),
    ("([:, 2])", (":", 2)),
    ({..., 3, ":"}, (..., 3, ":")),
]


def test_STypeLike_input_values_check_special_cases():
    assert isinstance(STypeLike, _STypeLike_Meta)
    a = STypeLike
    assert isinstance(a, STypeLike)


@pytest.mark.parametrize("in1", STypeLike_positiv_test_list)
def test_STypeLike_input_values_check_positive(in1):
    assert isinstance(in1, STypeLike)


@pytest.mark.parametrize("in1", STypeLike_negativ_test_list)
def test_STypeLike_input_values_check_negativ(in1):
    assert not isinstance(in1, STypeLike)


#
# Ending: STypeLike
#
# ############################################


# ############################################
#
# SType
# -----
#

SType_convert_positive_list = [
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
    ("{..., 3, :}", (..., 3, ":")),
]

# for acceptance as STypeLike values
SType_positiv_test_list = [
    ((3,)),
    ((":",)),
    ((...,)),
    ((3, 2)),
    ((":", 2)),
    ((..., 2)),
    ((3, ...)),
    ((":", ...)),
    ((..., 3, 10)),
    ((":", 3, 10)),
    ((..., ":", 10)),
    ((..., 3, ":")),
    ((..., 4, ...)),
]

SType_negativ_test_list = [
    ([3]),
    ("[3]"),
    ("(3)"),
    ("3"),
    ([":"]),
    ("[:]"),
    ("(:)"),
    (":"),
    ([...]),
    (...),
    ("..."),
    ("[...]"),
    ("(...)"),
    ([3, 2]),
    ([":", 2]),
    ([..., 2]),
    ("[..., 2]"),
    ("..., 2"),
    ([3, ...]),
    ("[3, ...]"),
    ("3, ..."),
    ([":", ...]),
    ("[:, ...]"),
    (":, ..."),
    ("(:, ...)"),
    ("[:, 2]"),
    ("(:, 2)"),
    (":, 2"),
    ([..., 3, 10]),
    ("[..., 3, 10]"),
    ("(..., 3, 10)"),
    ("..., 3, 10"),
    ([":", 3, 10]),
    ("(:, 3, 10)"),
    ("[:, 3, 10]"),
    (":, 3, 10"),
    ([..., ":", 10]),
    ("[..., :, 10]"),
    ("..., :, 10"),
    ("(..., :, 10)"),
    ([..., 3, ":"]),
    ("[..., 3, :]"),
    ("..., 3, :"),
    ("(..., 3, :)"),
    ("{..., 3, :}"),
    ("3,2"),
    ("[3, 2]"),
    ({"a": 3}),
    ("[3"),
    ("3)"),
    ("3.1"),
    (":]"),
    ("()"),
    ("[..]"),
    ("[....]"),
    ("[..., :, -10]"),
    ("...., :, 10"),
    ("(.., :, 10)"),
    ("([:, 2])", (":", 2)),
    ({..., 3, ":"}, (..., 3, ":")),
]


def test_SType_type():
    assert isinstance(SType, _SType_Meta)
    a = SType
    b = SType((2))
    isinstance(a, SType)
    isinstance(b, SType)


# check values against "SType"


@pytest.mark.parametrize("in1", SType_positiv_test_list)
def test_SType_check_instance(in1):
    assert isinstance(in1, SType)


@pytest.mark.parametrize("in1", SType_negativ_test_list)
def test_SType_check_not_instance(in1):
    assert not isinstance(in1, SType)


# check import cababilities (StypeLike used as parameter,
# result is tuple)


@pytest.mark.parametrize("in1, out1", SType_convert_positive_list)
def test_SType_check_result(in1, out1):
    assert SType(in1) == out1


# check of the sarray-test-cababilities

array_test_list = [
    ((3), np.array((1, 2, 3)), True),
    ([3], np.array((1, 2, 3)), True),
    ("[3]", np.array((1, 1, 2)), True),
    ("(3)", np.array(([1, 0], [1, 2], [2, 2])), False),
    ("3", np.array(([1, 2, 3, 4],)), False),
    ([":"], np.array((1,)), True),
    ((":"), np.array((1,)), True),
    ("[:]", np.array((1, 2, 3, 4, 5)), True),
    ("(:)", np.array(([1], [2])), False),
    (":", np.array(([1, 2], [2, 3])), False),
    ((...), np.array([]), True),
    ([...], np.array([]), True),
    (..., np.array([1, 2, 3]), True),
    ("...", np.array(([1], [2], [3])), True),
    ("[...]", np.array(([1, 2], [2, 2], [3, 3])), True),
    ("(...)", np.array(([1, 2], [2, 2], [3, 3])), True),
    ([3, 2], np.array(([1, 2], [2, 2], [2, 3])), True),
    ("3,2", np.array(([1, 2], [2, 2])), False),
    ("[3, 2]", np.array(([1], [2], [2])), False),
    ([":", 2], np.array(([1, 2], [2, 2])), True),
    ((":", 2), np.array(([1, 2, 3], [2, 2, 3])), False),
    ("[:, 2]", np.array(([1, 2],)), True),
    ("(:, 2)", np.array(([1, 2], [2, 2])), True),
    ((..., 2), np.array(([[1, 2], [2, 2]])), True),
    ([..., 2], np.array(([[1, 2], [2, 2]])), True),
    ("[..., 2]", np.array(([[1, 2, 3], [2, 2, 3]],)), False),
    ("..., 2", np.array((1, 2)), True),
    ((3, ...), np.array((3, 2, 1)), True),
    ([3, ...], np.array((3, 2, 1)), True),
    ("[3, ...]", np.array(((3, 2, 1), (3, 2, 1), (3, 2, 1))), True),
    ("3, ...", np.array(((3,), (3,), (3,))), True),
    (
        (":", ...),
        np.array((([3, 2], [1, 4]), ([3, 2], [1, 4]), ([3, 2], [1, 4]))),
        True,
    ),
    (
        [":", ...],
        np.array((([3, 2], [1, 4]), ([3, 2], [1, 4]), ([3, 2], [1, 4]))),
        True,
    ),
    ("[:, ...]", np.array([]), True),
    (":, ...", np.array([1, 2, 3]), True),
    ("(:, ...)", np.array([[1, 2, 3]]), True),
    ((..., 3, 1), np.array([[1], [2], [3]]), True),
    ([..., 3, 1], np.array([[1], [2], [3]]), True),
    ("[..., 3, 1]", np.array([1]), False),
    (
        "(..., 3, 1)",
        np.array(([[1], [2], [3]], [[1], [2], [3]], [[1], [2], [3]])),
        True,
    ),
    ("..., 3, 1", np.array(([[1]], [[2]], [[3]])), False),
    ((":", 3, 1), np.array(([[1], [2], [3]], [[1], [2], [3]])), True),
    ([":", 3, 1], np.array(([[1], [2], [3]], [[1], [2], [3]])), True),
    ("(:, 3, 1)", np.array(([[1], [2], [3]],)), True),
    ("[:, 3, 1]", np.array([[1], [2], [3]]), False),
    (":, 3, 1", np.array((([1]),)), False),
    ((..., ":", 1), np.array(([1], [2])), True),
    ([..., ":", 1], np.array(([1], [2])), True),
    (
        "[..., :, 1]",
        np.array(([[1], [2], [3]], [[1], [2], [3]], [[1], [2], [3]])),
        True,
    ),
    ("..., :, 1", np.array(([[1, 2, 3], [2, 2, 3]])), False),
    ("(..., :, 1)", np.array(([1], [2], [3])), True),
    ((..., 1), np.array([([1],)]), True),
    ([..., 1], np.array([([1],)]), True),
    ("[..., 1]", np.array([1]), True),
    ("..., 1", np.array([]), False),
    ("(..., 1)", np.array((1, 2)), False),
    (
        (..., 3, ":"),
        np.array(([[1], [2], [3]], [[1], [2], [3]], [[1], [2], [3]])),
        True,
    ),
    (
        [..., 3, ":"],
        np.array(([[1], [2], [3]], [[1], [2], [3]], [[1], [2], [3]])),
        True,
    ),
    ("[..., 3, :]", np.array(([[], [], []], [[], [], []], [[], [], []])), True),
    ("..., 3, :", np.array([1, 2, 3]), False),
    ("(..., 3, :)", np.array([[1], [2], [3]]), True),
]


@pytest.mark.parametrize("in1, in2, out1", array_test_list)
def test_SType_method_check_ndarray(in1, in2, out1):
    assert SType(in1).check_ndarray(in2) == out1


array_exception_test_list = [
    ("([:, 2])", np.array([])),
    ("(..., 3, ':')", np.array([1, 2, 3])),
    (("[..., :, '1']"), np.array(([[1], [2], [3]], [[1], [2], [3]], [[1], [2], [3]]))),
]


@pytest.mark.parametrize("in1, in2", array_exception_test_list)
def test_SType_method_check_ndarray_value_error_exception(in1, in2):
    with pytest.raises(ValueError):
        SType(in1).check_ndarray(in2)


#
# Ending: SType
#
# ############################################


# ############################################
#
# sndarray
# --------
#

array_test_list = [
    ([3], np.array((1, 2, 3)), True),
    ("[3]", np.array((1, 1, 2)), True),
    ("(3)", np.array(([1, 0], [1, 2], [2, 2])), False),
    ("3", np.array(([1, 2, 3, 4],)), False),
    ([":"], np.array((1,)), True),
    ("[:]", np.array((1, 2, 3, 4, 5)), True),
    ("(:)", np.array(([1], [2])), False),
    (":", np.array(([1, 2], [2, 3])), False),
    ([...], np.array([]), True),
    (..., np.array([1, 2, 3]), True),
    ("...", np.array(([1], [2], [3])), True),
    ("[...]", np.array(([1, 2], [2, 2], [3, 3])), True),
    ("(...)", np.array(([1, 2], [2, 2], [3, 3])), True),
    ([3, 2], np.array(([1, 2], [2, 2], [2, 3])), True),
    ("3,2", np.array(([1, 2], [2, 2])), False),
    ("[3, 2]", np.array(([1], [2], [2])), False),
    ([":", 2], np.array(([1, 2], [2, 2])), True),
    ((":", 2), np.array(([1, 2, 3], [2, 2, 3])), False),
    ("[:, 2]", np.array(([1, 2],)), True),
    ("(:, 2)", np.array(([1, 2], [2, 2])), True),
    ([..., 2], np.array(([[1, 2], [2, 2]])), True),
    ("[..., 2]", np.array(([[1, 2, 3], [2, 2, 3]],)), False),
    ("..., 2", np.array((1, 2)), True),
    ([3, ...], np.array((3, 2, 1)), True),
    ("[3, ...]", np.array(((3, 2, 1), (3, 2, 1), (3, 2, 1))), True),
    ("3, ...", np.array(((3,), (3,), (3,))), True),
    (
        [":", ...],
        np.array((([3, 2], [1, 4]), ([3, 2], [1, 4]), ([3, 2], [1, 4]))),
        True,
    ),
    ("[:, ...]", np.array([]), True),
    (":, ...", np.array([1, 2, 3]), True),
    ("(:, ...)", np.array([[1, 2, 3]]), True),
    ([..., 3, 1], np.array([[1], [2], [3]]), True),
    ("[..., 3, 1]", np.array([1]), False),
    (
        "(..., 3, 1)",
        np.array(([[1], [2], [3]], [[1], [2], [3]], [[1], [2], [3]])),
        True,
    ),
    ("..., 3, 1", np.array(([[1]], [[2]], [[3]])), False),
    ([":", 3, 1], np.array(([[1], [2], [3]], [[1], [2], [3]])), True),
    ("(:, 3, 1)", np.array(([[1], [2], [3]],)), True),
    ("[:, 3, 1]", np.array([[1], [2], [3]]), False),
    (":, 3, 1", np.array((([1]),)), False),
    ([..., ":", 1], np.array(([1], [2])), True),
    (
        "[..., :, 1]",
        np.array(([[1], [2], [3]], [[1], [2], [3]], [[1], [2], [3]])),
        True,
    ),
    ("..., :, 1", np.array(([[1, 2, 3], [2, 2, 3]])), False),
    ("(..., :, 1)", np.array(([1], [2], [3])), True),
    ([..., 1], np.array([([1],)]), True),
    ("[..., 1]", np.array([1]), True),
    ("..., 1", np.array([]), False),
    ("(..., 1)", np.array((1, 2)), False),
    (
        [..., 3, ":"],
        np.array(([[1], [2], [3]], [[1], [2], [3]], [[1], [2], [3]])),
        True,
    ),
    ("[..., 3, :]", np.array(([[], [], []], [[], [], []], [[], [], []])), True),
    ("..., 3, :", np.array([1, 2, 3]), False),
    ("(..., 3, :)", np.array([[1], [2], [3]]), True),
]

# we check if the import of stype (including converts) happens

array_import_test_list = []
for row in array_test_list:
    array_import_test_list.append((row[0], row[1]))


@pytest.mark.parametrize("in1, in2", array_import_test_list)
def test_sndarray_stype_setter_getter(in1, in2):
    a = sndarray(a=in2, stype=in1)
    assert a.stype == SType(in1)


# we check the check_stype method

@pytest.mark.parametrize("in1, in2, out1", array_test_list)
def test_sndarray_stype_check_type(in1, in2, out1):
    a = sndarray(a=in2, stype=in1)
    assert a.check_stype() == out1


#
# Ending: sndarray
#
# ############################################
