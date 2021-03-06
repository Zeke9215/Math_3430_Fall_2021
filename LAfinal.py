

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



#####################HW5#####################################Homework05



#1 Deleted Unstable QR

matrix_qr1 = [[1,1,1],[-1,1,0],[1,2,1]]
matrix_qr2 = [[1,1,3],[2,2,2],[2,2,-1]]





#2

def stable_QR(matrix:list) -> list:
    """A function that performs the stable Gram-Schmidt to return the QR Factorization of a given matrix.
    Args: A matrix stored as a list of lists.
    Returns: The QR factorization as a new matrix"""
    Q = [[0] * len(matrix[0]) for elements in matrix]
    v = [0 for elements in matrix]
    r = [[0] * len(matrix[0]) for elements in matrix]
    for j in range (len(matrix)):
        v[j] = matrix[j]
    for k in range(len(matrix)):
        r[k][k] = p_norm(v[k], 2)
        Q[k] = scalar_vector_multi(1 / r[k][k], v[k])
        for j in range(len(matrix)):
            r[j][k] = inner_product(Q[k],v[j])
            v[j] = add_vectors(v[j],scalar_vector_multi(-r[j][k],Q[k]))

    return [Q,r]

print("problem 2 stableQR########################################")
print(stable_QR(matrix_qr1))

print("""answer for matrix_qr1 should be [[[0.5773502691896258, 0.5773502691896258, 0.5773502691896258],
 [-0.7071067811865475, 0.7071067811865475, 0.0],
  [0.4082482904638618, 0.40824829046386235, -0.8164965809277268]],
 [[1.7320508075688772, 0, 0],
  [0.0, 1.4142135623730951, 0],
  [2.3094010767585034, 0.7071067811865475, 0.408248290463863]]]""")

print("input 2 -> matrix_qr2")
print(stable_QR(matrix_qr2))

print("""answer for matrix_qr2 should be [[[0.30151134457776363, 0.30151134457776363, 0.9045340337332909],
  [0.6396021490668313, 0.6396021490668313, -0.42640143271122105],
  [0.0, 0.0, 1.0]],
 [[3.3166247903554, 0, 0],
  [3.0151134457776365, 1.7056057308448835, 0],
  [0.30151134457776363, 2.984810028978546, 6.661338147750939e-16]]]""")



#####################################Homework06

#1 Delete unstable gs


#2 Orthonomal list return

def orthonormal_list_return(matrix:list) -> list:
    """A Function that returns an orthonormal list of vectors from a given list of vectors.
    Uses the Qr factorization but only returns Q.
        Args: A matrix stored as a list of lists.
        Returns: The list of orthonormal vectors as a new matrix"""
    Q = [[0] * len(matrix[0]) for elements in matrix]
    v = [0 for elements in matrix]
    r = [[0] * len(matrix[0]) for elements in matrix]
    for j in range(len(matrix)):
        v[j] = matrix[j]
    for k in range(len(matrix)):
        r[k][k] = p_norm(v[k], 2)
        Q[k] = scalar_vector_multi(1 / r[k][k], v[k])
        for j in range(len(matrix)):
            r[j][k] = inner_product(Q[k], v[j])
            v[j] = add_vectors(v[j], scalar_vector_multi(-r[j][k], Q[k]))

    return [Q]

print(orthonormal_list_return(matrix_qr1))

print("""answer for matrix_qr1 should be [[[0.5773502691896258, 0.5773502691896258, 0.5773502691896258],
 [-0.7071067811865475, 0.7071067811865475, 0.0],
  [0.4082482904638618, 0.40824829046386235, -0.8164965809277268]]""")

print("input 2 -> matrix_qr2")
print(orthonormal_list_return(matrix_qr2))

print("""answer for matrix_qr2 should be [[[0.30151134457776363, 0.30151134457776363, 0.9045340337332909],
  [0.6396021490668313, 0.6396021490668313, -0.42640143271122105],
  [0.0, 0.0, 1.0]]""")



######HOUSEHOLDER





def sign(x: float) ->float:
    """This function determines the sign of the function by checking if x is greater or less than 0
    input: x as a float
    output: -1 or 1"""
    if x >= 0:
        return 1
    else:
        return -1


def reflect_vector(vector_1: list)->list:
    """This function determines the reflection vector across an axis need to compute the householder algorithm
    input: a list as vector_1
    output: the reflected vector as a list v"""
    e = [0 for element in range(len(vector_1))]
    e[0] = 1
    addend = scalar_vector_multi(sign(vector_1[0])  * boolean_p_norm(vector_1),e)

    v = add_vectors(addend, vector_1)

    return v


def identity(size: int)->int:
    """this function returns an identity matrix of a given size
    input: an integer that determines the size of the identity matrix
    output: a matrix of the size of the int"""
    identity: list = [[0 for element in range(size)] for index in range(size)]
    for x in range(size):
        for y in range(size):
            identity[x][x] = 1
    return identity


def deep_copy(matrix_1: list) -> list:
    "this function takes Q from Stable Grahm schmidt"
    empty: list = [[0 for element in range(len(matrix_1[0]))] for index in range(len(matrix_1))]
    for x in range(len(matrix_1[0])):
        for y in range(len(matrix_1[0])):
            empty[x][y] = matrix_1[x][y]
    return empty


def conjugate_transpose(matrix_1: list) ->list:
    empty: list = [[0 for element in  range(len(matrix_1[0]))] for index in range(len(matrix_1))]
    empty_2: list = [[0 for element in range(len(matrix_1[0]))] for index in range(len(matrix_1))]
    for x in range(len(matrix_1[0])):
        for y in range(len(matrix_1[0])):
            empty[x][y] = (matrix_1[x][y].conjugate())
    for i in range(len(matrix_1[0])):
        for j in range(len(matrix_1)):
            empty_2[i][j] = empty[j][i]
    return empty_2



def v_v_multi(vector_1,vector_2):
    """this funtion does V*V component of F
    input: 2 vectors as a list
    output: V*V"""
    result = []
    vector_1 == vector_2
    for x in range(len(vector_1)):
        result.append(scalar_vector_multi(vector_1[x],vector_2))
    return result


def f_builder(vector_1: list) -> list:
    """This function returns the F component of the Householder algorithm """
    s = -2/(boolean_p_norm(vector_1))**2
    x = scalar_matrix_multi(s,v_v_multi(vector_1,vector_1))
    y = matrix_matrix_add(identity(len(vector_1)), x)
    return y


def Q_build(mtx :list, n: int):
    """This function builds Q for the Householder"""
    A: list = [[0 for j in range (n, len(mtx[i]))]for i in range(n,len(mtx))]
    for i in range(len(mtx)):
        for j in range(len(mtx[i])):
            if n+i < len(mtx[i]):
                if n+j < len(mtx[i]):
                    A[i][j] = mtx[n+i][n+j]
    v = reflect_vector(A[0])
    f = f_builder(v)
    Q = identity(len(mtx))
    for i in range(n,len(Q)):
        for j in range(n, len(Q)):
            Q[i][j] = f[i-n][j-n]
    return Q








def Householder(matrix_A: list) ->list:
    "This function takes a matrix as a lists of list and returns QR factorization using householder"
    R: list = deep_copy(matrix_A)
    Q_list: list = []
    for index in range(len(R)):
        Q_temp: list = Q_build(R,index)
        R = matrix_matrix_multi(Q_temp, R)
        Q_list.append(Q_temp)
    Q: list = Q_list[-1]
    Q: list = conjugate_transpose(Q_list[0])
    for index1 in range(1, len(Q_list)):
        ct = conjugate_transpose(Q_list[index1])
        Q = matrix_matrix_multi(Q, ct)
    return Q, R





print(Householder([[2,2,1],[-2,1,2],[18,0,0]]))

#print(sign(-3))
#print(reflect_vector([2,2,1]))
#print(identity(1))
#print(deep_copy([[2,2],[1,1]]))
#print(conjugate_transpose([[2,3],[5,2-3j]]))
#print(v_v_multi([5,2,1],[5,2,1]))
#print(f_builder([4.8, 2.4]))
#print(Q_build(f_builder([4.8, 2.4]), 1))




def backsub(matrix_a: list, vector_a:list) -> list:
    """This Function solves the upper triangular matrix and an input vector to return the solution vector, where
     Ax=B A is the upper triangular matrix and b is the input vector. x is the solution vector returned."""

    """Starting from the last column, take the diagonal and subtract the elements to the to the right multiplied by the solution
     and then proceed to divide by the diagonal."""

    """ARGS:
    matrix_a: A list of a list that represents the upper triangular matrix
    vector_a: An input vector stored as a list. """

    """Returns:
    Returns the solution vector."""

    result: list = [vector_a[-1]*(1/(matrix_a[-1][-1]))]
    for element in range(len(matrix_a) - 2, -1, -1):
        scal: float = vector_a[element]
        for index in range(len(result)):
            scal -=matrix_a[len(matrix_a)-1-index][element]*result[index]
        scal *=1/(matrix_a[element][element])
        result.append(scal)
    return result[: : -1]



def least_squares(matrix_a: list, vector_a: list) -> list:
    """Creates the least squares matrix from the input matrix an d vector."""

    """Least squares via QR factorization
    First we use householder to get the reduced QR factorization. Next we use the conjugate transpose function on Q 
    and set to a variable called Q_1. this is Q*b. Then we solve the Triangular system by multiplying the conjugate
     transpose of Q by vector A and calling it Q_2 Finally set the result to backsub(R,Q_2).
    """
    """ARGS:
    matrix_a: A matrix stored as a list of lists
    vector_a: a vector stored as a list"""

    """Returns:
    The least squares matrix"""


    Q, R = Householder(matrix_a)
    Q_1 = conjugate_transpose(Q)
    Q_2 = matrix_vector_multi(Q_1,vector_a)
    result = backsub(R, Q_2)
    return result

print("LS")
print(least_squares([[1,2,3],[1,1,2],[2,1,2]], [3,3,1]))



