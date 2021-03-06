

import LS
import pytest




bs_matrix_1 = [[2,1,3],[1,1,2],[2,2,1]]
bs_vector_1 = [4,4,2]

bs_matrix_2 = [[1,2,3],[2,2,1],[2,1,2]]
bs_vector_2 = [2,2,1]
def test_backsub():
    assert LS.backsub(bs_matrix_1,bs_vector_1) == [0.0, 0.0, 2.0]

def test_backsub():
    assert LS.backsub(bs_matrix_2,bs_vector_2) == [-0.5, 0.75, 0.5]

def test_least_squares():
    assert  LS.least_squares(bs_matrix_1,bs_vector_1) == [-4.62540056142711e-16, 7.195067539997726e-16, 2.0]

def test_least_squares():
    assert LS.least_squares(bs_matrix_2,bs_vector_2) == [1.1868783374443496e-16, 0.9999999999999997, 2.127886417069091e-16]