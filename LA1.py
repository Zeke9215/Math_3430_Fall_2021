







"""
This homework is due on 10/15/2021 by 11:59pm.


For this assignment you will be writing a python script to be named LA.py. In
this script you will need to write 6 functions. Every function must

1) Have a doc string.

2) Have type annotations

3) Be tested using unit testing.

Once you have finished writing LA.py you will upload it to the same github repo
you used for HW02. The functions you need to write are

#0 A function which takes as it's arguments two vectors stored as
lists and returns their sum, also stored as a list.


#1 A function which takes as it's arguments a vector stored as a list and a
scalar, and returns the scalar vector multiplication stored as a list.


#2 A function which takes as it's arguments a matrix, stored as a list of lists
where each component list represents a column of the matrix(you cannot represent
the matrix as a list of rows!) and a scalar and returns their product, also
stored as a list of lists where each component list represents a column. You
must use the function from problem #1. Failure to use this function will result
in an earned grade of 0.

#3 A function which takes as it's arguments two matrices stored as lists of
lists where each component list represents a column vector, and returns their
sum stored in the same manner. You must use the function in problem #0 in your
method here. Failure to use the function from problem #0 will reuslt in an
earned grade of 0.

#4 A function which takes as it's argument a matrix (stored as a list of lists,
each component list representing a column vector), and a vector stored as a
list, and returns the matrix-vector product. This function must compute the
matrix-vector product by calculating the neccessary linear combination of the
input matrices columns. All other methods of matrix-vector multiplication are
strictly forbidden and their use will result in a grade of 0. For this function
you must use the functions written for problem #0 and problem #1. Failure to use
these functions will result in an earned grade of 0.

#5 A function which takes as it's arguments two matrices, each stored as a list
of lists where each component list represents a column vector, and returns their
product stored in the same manner. To earn any credit on this problem you must
use the function from problem #4 to implement the matrix-vector method of
matrix-matrix multiplication. Use of any other method will result in an earned
grade of 0.
"""


# Begin Example
# Problem #0

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
"Multiplies a vector stored as a list by a scalar, and returns the scalar-vector multiplication as a list. "


vector_a = [1,2,3]
scalar_a = 3

vector_q = [3,3,3]
scalar_q = 10

def scalar_vector_multi(scalar_a:float,vector_a:list)->list:
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

def scalar_vector_multi(scalar_q:float,vector_q:list)->list:
    result = [0 for elements in vector_q]
    for index in range(len(vector_q)):
        result[index] = scalar_q * vector_q[index]
    return result



print("problem#1")
print(scalar_vector_multi(scalar_a,vector_a))
print("answer for Q1 should be [3,6,9]")
print("test input#2 for problem1")
print(scalar_vector_multi(scalar_q,vector_q))
print("the answer should be [30,30,30]")


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


def scalar_matrix_multi(scalar_w:float,matrix_w:list)->list:
    result = [0 for elements in matrix_w]
    for index in range(len(matrix_w)):
        result[index] = scalar_vector_multi(scalar_w,matrix_w[index])  #function from problem1
    return result

print("problem#2")
print((scalar_matrix_multi(scalar_b,matrix_a)))
print("answer should be [[10,5,10],[15,5,15],[25,25,25] ")
print("test input #2 for problem2")
print(scalar_matrix_multi(scalar_w,matrix_w))
print("the answer should be [[2,4,6],[6,4,1],[10,10,10]")



#3  ######################################################################################################

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
    result = [0 for elements in matrix_b]
    for index in range(len(matrix_b)):
        result[index] = add_vectors(matrix_b[index], matrix_c[index]) #function from problem 0
    return result


def matrix_matrix_add(matrix_u:list, matrix_v:list)->list:
    result = [0 for elements in matrix_u]
    for index in range(len(matrix_u)):
        result[index] = add_vectors(matrix_u[index], matrix_v[index]) #function from problem 0
    return result


print("Problem#3")
print(matrix_matrix_add(matrix_b, matrix_c))
print("answer should be [[11,11,11],[3,3,3],[10,10,10]]")

print("test input#2 for problem 3")
print(matrix_matrix_add(matrix_u, matrix_v))
print("answer should be [[8,9,10],[5,4,4],[3,3,4] ")

#4  ######################################################################################

matrix_d =[[10,10,10],[7,7,7],[15,13,15]]
vector_c = [7,8,9]

matrix_m =[[1,1,1],[9,8,7],[3,4,3]]
vector_n = [6,7,8]

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
        result = [scalar_vector_multi(matrix_d[0],vector_c[0]), scalar_vector_multi(matrix_d[1],vector_c[1]),scalar_vector_multi(matrix_d[2],vector_c[2])]
        result = add_vectors(result[0], result[1])
        result = add_vectors(result, scalar_vector_multi(matrix_d[2],vector_c[2]))

        return result

def matrix_vector_multi(matrix_m:list,vector_n:list)->list:
    result = [0 for elements in matrix_m]
    for index in range(len(matrix_m)):
        result = [scalar_vector_multi(matrix_m[0],vector_n[0]), scalar_vector_multi(matrix_m[1],vector_n[1]), scalar_vector_multi(matrix_m[2],vector_n[2])]
        result = add_vectors(result[0], result[1])
        result = add_vectors(result, scalar_vector_multi(matrix_m[2],vector_n[2]))

        return result



print("Problem#4")
print (matrix_vector_multi(vector_c, matrix_d))
print('answer should be [261,243,261]')
print("test input#2 for problem 4")
print(matrix_vector_multi(vector_n,matrix_m))
print("the answer should be [93,94,79]")



#5 Matrix-Matrix Multiplication ######################################################


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
matrix_e = [[3,5,3],[4,8,4],[1,4,1]]
matrix_f = [[2,1,5],[9,2,1],[1,2,7]]

matrix_g = [[1,1,1],[9,9,7],[2,2,1]]
matrix_h = [[0,3,3],[2,3,4],[5,5,5]]

def matrix_matrix_multi(matrix_f:list,matrix_e:list)->list:
    result = [matrix_f]
    for index in range(len(matrix_f)):
        result[index]  = matrix_vector_multi(matrix_f[0],matrix_e), matrix_vector_multi(matrix_f[1],matrix_e), matrix_vector_multi(matrix_f[2],matrix_e)
        return result

def matrix_matrix_multi(matrix_h:list,matrix_g:list)->list:
    result = [matrix_h]
    for index in range(len(matrix_h)):
        result[index]  = matrix_vector_multi(matrix_h[0],matrix_g), matrix_vector_multi(matrix_h[1],matrix_g), matrix_vector_multi(matrix_h[2],matrix_g)
        return result




print("Problem#5")
print("E x F")
print(matrix_matrix_multi(matrix_f,matrix_e))
print("the answer should be [[15,38,15],[36,65,36],[18,48,18]")

print("Test input #2 for problem5")
print("G x H")
print(matrix_matrix_multi(matrix_h,matrix_g))
print("the answer should be [[33,33,24],[37,37,27],[60,60,45]] ")


#######################################################################################################################################################################



#hw4

#Problem1

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
print("1. Test input 1 should be 13.453624..")
print('1. Test input 2 should be 9')

#Problem2
"""A function that returns the P-Norm of a vector. Runs the absolute value function on the vector and 
   raise to the exponent of the scalar, sums the elements, then raises to the 1/scalar
   Args: A vector stored as a List, A scalar stored as a float
   Returns: The p-norm as a float """
def p_norm(vector: list, scalar: float)->float:
    result = 0
    for index in range(len(vector)):
        y = (abs_value(vector[index]))**scalar
        result = result + y
    result = result**(1/scalar)
    return result

print(p_norm([1,2,3],2))
print(p_norm([-1,2,-3],3))
print("2. test input 1 should be 3.741657..")
print("2. test input 2 should be 3.301927...")


#problem 3
"""A function that finds the infinity norm which is the max value
   Args: Vector stored as a list
   Returns: thr infinity norm returned as a float"""
def infinity_norm(vector: list)-> float:
    result = []
    for index in range(len(vector)):
        result.append(abs_value(vector[index]))


    result = max(result)
    return result

print(infinity_norm([1,2+5j,3]))
print(infinity_norm([-20,2,3]))
print("3. infinity norm should be 5.385164..")
print("3. infinity norm should be 20")

#problem4
"""A function that returns the p norm or a boolean value depending on the inputs
   Args: Vector stored as a list, Boolean value as False as default, and a scalar stored as a float, default 2
    Returns:: the p norm as a float"""
def boolean_p_norm(vector: list, boolean: bool = False, scalar: float = 2 )->float:
    if boolean ==True:
        x = infinity_norm(vector)
    else:
        x = p_norm(vector,scalar)
    return x

print(boolean_p_norm([4,2,7]))
print(boolean_p_norm([5,5,5]))
print("4.test input 1 should be 8.3066..")
print("4. test input 2 should be 8.66025..")

#problem 5
"""A function that returns the inner product of two vectors
   Args: Vector a and Vector b stored as lists
   Returns: the inner product returned as a float"""
def inner_product(vector_a: list, vector_b: list)->float:
    result = 0
    for index in range(len(vector_a)):
        y = vector_a[index]*vector_b[index]
        result = result + y
    return result

print(inner_product([1,2,5+6j],[4,3,5+6j]))
print(inner_product([1,1,1],[3,4,5]))
print("5. inner product should be -1+60j")
print("5. inner product 2 should be 12")
