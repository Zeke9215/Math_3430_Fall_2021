import LA


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



