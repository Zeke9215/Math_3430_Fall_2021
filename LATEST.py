
import pytest
import LA

scalar_01 = 2

test_vector_01 =[1,2]
test_vector_02 = [3,3]
test_vector_03 = [5,4]

test_matrix_01 = [[3,2],[1,2]]
test_matrix_02 = [[5,5],[5,9]]
test_matrix_03 = [[7,8],[8,2]]

test_matrix_04 = [[1,2,3],[2,2,2],[8,8,8]]
test_matrix_05 = [[3,3,3],[1,1,1],[5,5,5]]


def test_add_vectors_01():
    assert LA.add_vectors(test_vector_01,test_vector_02) == [4,5]




def test_add_vectors_02():
    assert LA.add_vectors(test_vector_02,test_vector_03) == [8,7]



##############################################################################################################

def test_scalar_vector_multi_01():
    assert LA.scalar_vector_multi(scalar_01 ,test_vector_01) == [2,4]




def test_scalar_vector_multi_02():
    assert LA.scalar_vector_multi(scalar_01,test_vector_02) == [6,6]



#########################################################################################################

def test_scalar_matrix_multi_01():
    assert LA.scalar_matrix_multi(scalar_01,test_matrix_01) == [[6,4],[2,4]]


def test_scalar_matrix_multi_02():
    assert LA.scalar_matrix_multi(scalar_01,test_matrix_02) == [[10,10],[10,18]]


######################################################################################



def test_matrix_addition_01():
    assert LA.matrix_matrix_add(test_matrix_01,test_matrix_02) == [[8,7],[6,11]]

def test_matrix_addition_02():
    assert LA.matrix_matrix_add(test_matrix_02,test_matrix_03) == [[12,13],[13,11]]

#########################################################################################




def test_matrix_vector_multi_01():
    assert LA.matrix_vector_multi(test_matrix_01,test_vector_01) == [5,6]

def test_matrix_vector_multi_02():
    assert LA.matrix_vector_multi(test_matrix_02,test_vector_02) == [30,42]

#############################################

def test_matrix_matrix_multi_01():
    assert LA.matrix_matrix_multi(test_matrix_04,test_matrix_05) == [[33,36,39],[11,12,13],[55,60,65]]

def test_matrix_matrix_multi_02():
    assert LA.matrix_matrix_multi(test_matrix_02,test_matrix_03) == [[75,107],[50,58]]






scalar_a = -9
scalar_b = -15
p_norm_vector_01 = [1,2,3]
p_norm_scalar_01 = 2
p_norm_vector_02 = [-1,2,-3]
p_norm_scalar_02 = 3
infinity_vector_01 = [1,2+5j,3]
infinity_vector_02 = [-20,2,3]
boolean_p_norm_vector_01 = [4,2,5]
boolean_p_norm_vector_02 = [3,5,6]
inner_product_vector_01 = [1,2,5+6j]
inner_product_vector_02 = [4,3,5+6j]
inner_product_vector_03 = [1,5,7]
inner_product_vector_04 = [3,5,6+3j]





def test_abs_value_01():
    print(LA.abs_value(scalar_a))
    assert LA.abs_value(scalar_a) == 9.0

def test_abs_value_02():
    assert LA.abs_value(scalar_b) == 15.0

def test_p_norm_01():
    assert LA.p_norm(p_norm_vector_01,p_norm_scalar_01) == 3.7416573867739413

def test_p_norm_02():
    assert LA.p_norm(p_norm_vector_02,p_norm_scalar_02) == 3.3019272488946263

def test_infinity_norm_01():
    assert LA.infinity_norm(infinity_vector_01) == 5.385164807134504

def test_infinity_norm_02():
    assert LA.infinity_norm(infinity_vector_02) == 20.0

def test_boolean_p_norm_01():
    assert LA.boolean_p_norm(boolean_p_norm_vector_01) == 6.708203932499369

def test_boolean_p_norm_02():
    assert LA.boolean_p_norm(boolean_p_norm_vector_02) == 8.366600265340756

def test_inner_product_01():
    assert LA.inner_product(inner_product_vector_01,inner_product_vector_02) == (-1+60j)

def test_inner_product_02():
    assert LA.inner_product(inner_product_vector_03,inner_product_vector_04) == (70+21j)
