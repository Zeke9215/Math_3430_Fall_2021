import pytest
import LA1

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
    assert LA1.add_vectors(test_vector_01,test_vector_02) == [4,5]




def test_add_vectors_02():
    assert LA1.add_vectors(test_vector_02,test_vector_03) == [8,7]



#############################################################################################################

def test_scalar_vector_multi_01():
    assert LA1.scalar_vector_multi(scalar_01 ,test_vector_01) == [2,4]




def test_scalar_vector_multi_02():
    assert LA1.scalar_vector_multi(scalar_01,test_vector_02) == [6,6]



#########################################################################################################

def test_scalar_matrix_multi_01():
    assert LA1.scalar_matrix_multi(scalar_01,test_matrix_01) == [[6,4],[2,4]]


def test_scalar_matrix_multi_02():
    assert LA1.scalar_matrix_multi(scalar_01,test_matrix_02) == [[10,10],[10,18]]


######################################################################################



def test_matrix_addition_01():
    assert LA1.matrix_matrix_add(test_matrix_01,test_matrix_02) == [[8,7],[6,11]]

def test_matrix_addition_02():
    assert LA1.matrix_matrix_add(test_matrix_02,test_matrix_03) == [[12,13],[13,11]]

#########################################################################################




def test_matrix_vector_multi_01():
    assert LA1.matrix_vector_multi(test_matrix_01,test_vector_01) == [5,6]

def test_matrix_vector_multi_02():
    assert LA1.matrix_vector_multi(test_matrix_02,test_vector_02) == [30,42]

#############################################

def test_matrix_matrix_multi_01():
    assert LA1.matrix_matrix_multi(test_matrix_04,test_matrix_05) == [[33,36,39],[11,12,13],[55,60,65]]

def test_matrix_matrix_multi_02():
    assert LA1.matrix_matrix_multi(test_matrix_02,test_matrix_03) == [[75,107],[50,58]]



####I believe the last two tests failed because there might be an index error.
###I may have gotten the arguements mixed up when defining matrix_vector_multiplication