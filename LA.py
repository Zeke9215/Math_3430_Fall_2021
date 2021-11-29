#########0
def add_vectors(vector_a: list[float],
                vector_b: list[float]) -> list[float]:
    """Adds the two input vectors.

    Creates a result vector stored as a list of 0's the same length as the input
    then overwrites each element of the result vector with the corresponding
    element of the sum of the input vectors. Achieves this using a for loop over
    the indices of result.

    Args:
        vector_a: A vector stored as a list.
        vector_b: A vector, the same length as vector_a, stored as a list.

    Returns:
       The sum of the input vectors stored as a list.
    """
    result: list[float] = [0 for element in vector_a]
    for index in range(len(result)):
        result[index] = vector_a[index] + vector_b[index]
    return result

# End Example
# Note that you must add unit tests for problem 0!!!!!


#1. Scalar Vector Multiplication  #########################################################################

vector_a = [1,2,3]
scalar_a = 3

vector_q = [3,3,3]
scalar_q = 10

def scalar_vector_multi(scalar_a:float, vector_a:list)->list:
    "Multiplies the vector stored as a list by a scalar"
    "Creates a vector of zeros in vector a to be overwritten by the result"
    "The result is the elements in the list multiplied by the scalar"

    "Args:"
    "input#1"
    "Vector_a stored as a list"
    "scalar_a stored as a scalar"
    "input#2"
    "vector_q stored as a list"
    "scalar_q stored as a list"
    "scalar_vector_multi(scalar_a,vector_a) multiplies vector stored as a list by scalar"

    "returns:"
    "input#1"
    "the elements in vector_a multiplied by a scalar_a returned as a new list"
    "input#2"
    "the elements in vector_q multiplied by scalar_q returned as a new list"
    result = [0 for elements in vector_a]
    for index in range(len(vector_a)):
        result[index] = scalar_a * vector_a[index]
    return result



print(scalar_vector_multi(scalar_a,vector_a))


#2.Matrix-Scalar Multiplication #############################################################################


matrix_a = [[2,1,2,],[3,1,3],[5,5,5]]
scalar_b = 5

matrix_w = [[1,2,3],[3,2,1],[5,5,5]]
scalar_w = 2


def scalar_matrix_multi(scalar_b:float,matrix_a:list)->list:
    "Multiplies the Matrix stored as a list of a list by a scalar"
    "Creates a vector of zeros in the Matrix to be overwritten by the result"
    "The result is the elements in the matrix multiplied by the scalar"
    "use scalar_vector_multi from problem 1 to multiply the list elements in the matrix by the scalar"

    "Args:"
    "input#1"
    "matrix_a stored as a list of a list"
    "scalar_b stored as a scalar"
    "input#2"
    "matrix_w stored as a list of a list"
    "scalar_w stored as a scalar"
    "scalar_vector_multi(scalar_a,vector_a) multiplies vector stored as a list by scalar"

    "returns:"
    "input#1"
    "the elements in matrix_a multiplied by a scalar_b returned as a new matrix(list of lists)"
    "input#2"
    "the elements in matrix_w multiplied by scalar_w returned as a new matrix"
    result = [0 for elements in matrix_a]
    for index in range(len(matrix_a)):
        result[index] = scalar_vector_multi(scalar_b,matrix_a[index]) #function from Problem1
    return result


#3  ###########

matrix_b = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
matrix_c = [[10, 1, 10], [1, 1, 1], [7, 7, 7]]

matrix_u = [[6, 7, 8], [4, 4, 4], [3, 3, 3]]
matrix_v = [[2, 2, 2], [1, 0, 0], [0, 0, 1]]


def matrix_matrix_add(matrix_b:list, matrix_c:list)->list:
    "Adds two matrices and returns their sum"
    "Stores two matrices as a list of lists. Uses the function add_vectors from problem1 to add corresponding elements"
    "The result is the added elements as a new matrix"

    "Args:"
    "input#1"
    "matrix_b stored as a list of lists"
    "matrix_c stored as a list of lists"
    "input#2"
    "matrix_u stored as a list of lists"
    "matrix_v stored as a list of lists"
    "add_vectors(vector,vector) adds the corresponding vectors in each matrix with their corresponding elements and returns the sum"

    "returns:"
    "input#1"
    "the elements in matrix_b added to the elements in matrix_c and return as a new matrix"
    "input#2"
    "the elements in matrix_u added to the elements in matrix_v and returned as a new matrix"
    result = [0 for element in matrix_b]
    for index in range(len(matrix_b)):
        result[index] = add_vectors(matrix_b[index], matrix_c[index]) #function from problem 0
    return result





#4  ######

matrix_d =[[10,10,10],[7,7,7],[1,1,1]]
vector_c = [1,0,2]



def matrix_vector_multi(matrix_d:list,vector_c:list)->list:
    "Multiplies a matrix stored as a list of lists by a vector stored as a list"
    "Multiplies the elements in the matrix by the corresponding vector elements"
    "The result is the new list of the product of the matrix and vector"
    "first use scalar_vector_multi to multiply to the corresponding elements of the matrix to the vector"
    "then use add_vectors to add the multiplied elements to for a new list."

    "Args:"
    "input#1"
    "matrix_d stored as a list of lists"
    "vector_c stored as a list"
    "input#2"
    "matrix_m stored as a list of list"
    "vector_n stored as a list"

    "returns:"
    "input#1"
    "the elements in matrix_d multiplied by a vector_c returned as a new list"
    "input#2"
    "the elements in matrix_m multiplied by vector_n returned as a new list"
    result = [0 for elements in matrix_d]
    for index in range(len(matrix_d)):

        result = add_vectors(result,scalar_vector_multi(vector_c[index],matrix_d[index]))

    return result



print("Problem#4")
print (matrix_vector_multi(matrix_d, vector_c))


#55555555555555555555555

matrix_a1 = [[2,8],[7,3]]
matrix_b1 = [[1,4],[33,5]]

def matrix_matrix_multi(matrix_a1:list,matrix_b1:list)->list:
    "Multiplies the vector stored as a list by a scalar"
    "Creates a vector of zeros in vector a to be overwritten by the result"
    "The result is the elements in the list multiplied by the scalar"

    "Args:"
    "input#1"
    "Vector_a stored as a list"
    "scalar_a stored as a scalar"
    "input#2"
    "vector_q stored as a list"
    "scalar_q stored as a list"
    "scalar_vector_multi(scalar_a,vector_a) multiplies vector stored as a list by scalar"

    "returns:"
    "input#1"
    "the elements in vector_a multiplied by a scalar_a returned as a new list"
    "input#2"
    "the elements in vector_q multiplied by scalar_q returned as a new list"
    result = [0 for elements in matrix_a1]
    for index in range(len(matrix_a1)):
        result[index] = matrix_vector_multi(matrix_a1,matrix_b1[index])
    return result

print(matrix_b1)
print(matrix_b1[1])
print(matrix_b1[1][0])
print(matrix_matrix_multi(matrix_a1,matrix_b1))


#######################################################################################homework04

#1 absval

def abs_value(scalar: complex)-> float:
    """A function that returns the absolute value of real and complex numbers
    Args: A complex or real number
    Returns: The absolute value of the input as a float
    """
    z = scalar.conjugate()
    x = (scalar*z)**(1/2)
    return x.real
print(abs_value(9+10j))
print(abs_value(-9))


####2
def p_norm(vector: list, scalar: float)->float:
    """A function that returns the P-Norm of a vector. Runs the absolute value function on the vector and
       raise to the exponent of the scalar, sums the elements, then raises to the 1/scalar
       Args: A vector stored as a List, A scalar stored as a float
       Returns: The p-norm as a float """
    result = 0
    for index in range(len(vector)):
        y = (abs_value(vector[index]))**scalar
        result = result + y
    result = result**(1/scalar)
    return result

print(p_norm([1,2,3],2))
print(p_norm([-1,2,-3],3))


###3
def infinity_norm(vector: list)-> float:
    """A function that finds the infinity norm which is the max value
       Args: Vector stored as a list
       Returns: thr infinity norm returned as a float"""
    result = []
    for index in range(len(vector)):
        result.append(abs_value(vector[index]))


    result = max(result)
    return result

print(infinity_norm([1,2+5j,3]))
print(infinity_norm([-20,2,3]))


####4
def boolean_p_norm(vector: list, boolean: bool = False, scalar: float = 2 )->float:
    """A function that returns the p norm or a boolean value depending on the inputs
       Args: Vector stored as a list, Boolean value as False as default, and a scalar stored as a float, default 2
        Returns:: the p norm as a float"""
    if boolean ==True:
        x = infinity_norm(vector)
    else:
        x = p_norm(vector,scalar)
    return x

print(boolean_p_norm([4,2,7]))




#problem 5

def inner_product(vector_a: list, vector_b: list)->float:
    """A function that returns the inner product of two vectors
       Args: Vector a and Vector b stored as lists
       Returns: the inner product returned as a float"""
    result = 0
    for index in range(len(vector_a)):
        y = vector_a[index]*vector_b[index]
        result = result + y
    return result

print(inner_product([1,2,5+6j],[4,3,5+6j]))


def p_norm(vector: list, scalar: float ) ->float:
    result = 0
    for index in range(len(vector)):
        y = (abs_value(vector[index]))**scalar
        result = result + y
    result = result**(1/scalar)
    return result

##print(p_norm([1,2,3],2))
##print(p_norm([-1,2,-3],3))

