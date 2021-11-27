def add_vectors(vector_a: list[float],
                vector_b: list[float]) -> list[float]:

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

def scalar_vector_multi(scalar_a:float, vector_a:list)->list:

    result = [0 for elements in vector_a]
    for index in range(len(vector_a)):
        result[index] = scalar_a * vector_a[index]
    return result






#2.Matrix-Scalar Multiplication #############################################################################


matrix_a = [[2,1,2,],[3,1,3],[5,5,5]]
scalar_b = 5

matrix_w = [[1,2,3],[3,2,1],[5,5,5]]
scalar_w = 2


def scalar_matrix_multi(scalar_b:float,matrix_a:list)->list:

    result = [0 for elements in matrix_a]
    for index in range(len(matrix_a)):
        result[index] = scalar_vector_multi(scalar_b,matrix_a[index]) #function from Problem1
    return result




#3  ######################################################################################################

matrix_b = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
matrix_c = [[10, 1, 10], [1, 1, 1], [7, 7, 7]]

matrix_u = [[6, 7, 8], [4, 4, 4], [3, 3, 3]]
matrix_v = [[2, 2, 2], [1, 0, 0], [0, 0, 1]]


def matrix_matrix_add(matrix_b:list, matrix_c:list)->list:
    result = [0 for element in matrix_b]
    for index in range(len(matrix_b)):
        result[index] = add_vectors(matrix_b[index], matrix_c[index]) #function from problem 0
    return result





#4  ######################################################################################

matrix_d =[[10,10,10],[7,7,7],[1,1,1]]
vector_c = [1,0,2]



def matrix_vector_multi(matrix_d:list,vector_c:list)->list:

    result = [0 for elements in matrix_d]
    for index in range(len(matrix_d)):

        result = add_vectors(result,scalar_vector_multi(vector_c[index],matrix_d[index]))

    return result





#55555555555555555555555

matrix_a1 = [[2,8],[7,3]]
matrix_b1 = [[1,4],[33,5]]

def matrix_matrix_multi(matrix_a1:list,matrix_b1:list)->list:
    result = [0 for elements in matrix_a1]
    for index in range(len(matrix_a1)):
        result[index] = matrix_vector_multi(matrix_a1,matrix_b1[index])
    return result



#####################################################################################################################

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




def p_norm(vector: list, scalar: float ) ->float:
    result = 0
    for index in range(len(vector)):
        y = (abs_value(vector[index]))**scalar
        result = result + y
    result = result**(1/scalar)
    return result

##print(p_norm([1,2,3],2))
##print(p_norm([-1,2,-3],3))

def boolean_p_norm(vector: list, boolean: bool = False, scalar: float = 2 )->float:
    """A function that returns the p norm or a boolean value depending on the inputs
       Args: Vector stored as a list, Boolean value as False as default, and a scalar stored as a float, default 2
        Returns:: the p norm as a float"""
    if boolean ==True:
        x = infinity_norm(vector)
    else:
        x = p_norm(vector,scalar)
    return x



def abs_value(scalar: complex)-> float:
    """A function that returns the absolute value of real and complex numbers
    Args: A complex or real number
    Returns: The absolute value of the input as a float
    """
    z = scalar.conjugate()
    x = (scalar*z)**(1/2)
    return x.real
#print(abs_value(9+10j))
#print(abs_value(-9))








#1

matrix_qr1 = [[1,1,1],[-1,1,0],[1,2,1]]

def unstable_QR(matrix:list) -> list:
    Q = [0 for elements in matrix]
    v = [0 for elements in matrix]
    r = [[0] * len(matrix[0]) for elements in matrix]
    for j in range (len(matrix)):
        v[j] = matrix[j]
        for k in range(j):
            r[j][k] = inner_product(Q[k],v[j])
            v[j] = add_vectors(v[j],scalar_vector_multi(-r[j][k],Q[k]))
        r[j][j] = p_norm(v[j],2)
        Q[j] = scalar_vector_multi(1/r[j][j],v[j])
    return [Q,r]





#2

def stable_QR(matrix:list) -> list:
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


#print(stable_QR(matrix_qr1))

###################################################################################################################



####Homework6

#1 Orthonomal list return

def orthonormal_list_return(matrix:list) -> list:

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

#print(orthonormal_list_return(matrix_qr1))


####################################################################################################################










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















































