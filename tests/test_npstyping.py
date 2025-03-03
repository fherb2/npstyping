import numpy as np
import pytest
from npstyping import Colon, STypeLike, SType, sndarray


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
