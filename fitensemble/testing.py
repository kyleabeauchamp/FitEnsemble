import os
import functools
import numpy as np
from numpy.testing import (assert_allclose, assert_almost_equal,
  assert_approx_equal, assert_array_almost_equal, assert_array_almost_equal_nulp,
  assert_array_equal, assert_array_less, assert_array_max_ulp, assert_equal,
  assert_raises, assert_string_equal, assert_warns)
from nose.tools import ok_, eq_, raises
from nose import SkipTest
from pkg_resources import resource_filename
from scipy.sparse import isspmatrix

def eq(o1, o2, decimal=6, err_msg=''):
    assert (type(o1) is type(o2)), 'o1 and o2 not the same type: %s %s' % (type(o1), type(o2))

    if isinstance(o1, dict):
        assert_dict_equal(o1, o1, decimal)
    elif isinstance(o1, float):
        np.testing.assert_almost_equal(o1, o2, decimal)
    elif isspmatrix(o1):
        assert_spase_matrix_equal(o1, o1, decimal)
    elif isinstance(o1, np.ndarray):
        if o1.dtype.kind == 'f' or o2.dtype.kind == 'f':
            # compare floats for almost equality
            assert_array_almost_equal(o1, o2, decimal, err_msg=err_msg)
        elif o1.dtype.type == np.core.records.record:
            # if its a record array, we need to comparse each term
            assert o1.dtype.names == o2.dtype.names
            for name in o1.dtype.names:
                eq(o1[name], o2[name], decimal=decimal, err_msg=err_msg)
        else:
            # compare everything else (ints, bools) for absolute equality
            assert_array_equal(o1, o2, err_msg=err_msg)
    # probably these are other specialized types
    # that need a special check?
    else:
        eq_(o1, o2)
