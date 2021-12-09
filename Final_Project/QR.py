import LA



#2

def stable_QR(matrix:list) -> list:
    """A function that performs the stable Gram-Schmidt to return the QR Factorization of a given matrix.
    Args: A matrix stored as a list of lists.
    Returns: The QR factorization as a new matrix"""
    Q: list[complex] = [[0] * len(matrix[0]) for elements in matrix]
    v: list[complex] = [0 for elements in matrix]
    r: list[complex] = [[0] * len(matrix[0]) for elements in matrix]
    for j in range (len(matrix)):
        v[j] = matrix[j]
    for k in range(len(matrix)):
        r[k][k] = LA.p_norm(v[k], 2)
        Q[k] = LA.scalar_vector_multi(1 / r[k][k], v[k])
        for j in range(len(matrix)):
            r[j][k] = LA.inner_product(Q[k],v[j])
            v[j] = LA.add_vectors(v[j],LA.scalar_vector_multi(-r[j][k],Q[k]))

    return [Q,r]


#2 Orthonomal list return

def orthonormal_list_return(matrix:list) -> list:
    """A Function that returns an orthonormal list of vectors from a given list of vectors.
    Uses the Qr factorization but only returns Q.
        Args: A matrix stored as a list of lists.
        Returns: The list of orthonormal vectors as a new matrix"""
    Q: list[complex] = [[0] * len(matrix[0]) for elements in matrix]
    v: list[complex] = [0 for elements in matrix]
    r: list[complex] = [[0] * len(matrix[0]) for elements in matrix]
    for j in range(len(matrix)):
        v[j] = matrix[j]
    for k in range(len(matrix)):
        r[k][k] = LA.p_norm(v[k], 2)
        Q[k] = LA.scalar_vector_multi(1 / r[k][k], v[k])
        for j in range(len(matrix)):
            r[j][k] = LA.inner_product(Q[k], v[j])
            v[j] = LA.add_vectors(v[j], LA.scalar_vector_multi(-r[j][k], Q[k]))

    return [Q]





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
    e: list[complex] = [0 for element in range(len(vector_1))]
    e[0] = 1
    addend: list[complex] = LA.scalar_vector_multi(sign(vector_1[0])  * LA.boolean_p_norm(vector_1),e)

    v: list[complex] = LA.add_vectors(addend, vector_1)

    return v


def identity(size: int)->list:
    """this function returns an identity matrix of a given size
    input: an integer that determines the size of the identity matrix
    output: a matrix of the size of the int"""
    identity: list = [[0 for element in range(size)] for index in range(size)]
    for x in range(size):
        for y in range(size):
            identity[x][x] = 1
    return identity


def deep_copy(matrix_1: list) -> list:
    """Makes a deep copy of input matrix
    This function makes a deep copy of the input matrix as a list
    
    Args:
        matrix_1: input matrix
    
    Returns: A deep compy of the input matrix
    """
    
    
    empty: list = [[0 for element in range(len(matrix_1[0]))] for index in range(len(matrix_1))]
    for x in range(len(matrix_1[0])):
        for y in range(len(matrix_1[0])):
            empty[x][y] = matrix_1[x][y]
    return empty


def conjugate_transpose(matrix_1: list) ->list:
    """This function returns the conjugate transpose needed for the householder algorithm
    Args:
        matrix_1: the input matrix
    Returns:
        The conjugate transpose of the input matrix as a list.
    
    """
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
    output: V*V needed for the component of F needed for Householder"""
    result: list = []
    #vector_1 == vector_2
    for x in range(len(vector_1)):
        result.append(LA.scalar_vector_multi(vector_1[x], vector_2))
    return result


def f_builder(vector_1: list) -> list:
    """This function returns the F component of the Householder algorithm
    Builds the F component of the householder matrix
    
    Args:
        vector_1: input vector as a list
    
    Returns:
        y: F component as a list
    """
    s = -2/(LA.boolean_p_norm(vector_1))**2
    x = LA.scalar_matrix_multi(s,v_v_multi(vector_1,vector_1))
    y = LA.matrix_matrix_add(identity(len(vector_1)), x)
    return y


def Q_build(mtx :list, n: int):
    """This function builds Q for the Householder
    Builds the Q component by the Householder matrix algorithm
    
    Args:
        mtx: input matrix as a list
        n: integer 
    Returns:
        Q: the q component of the householder matrix as a list.
    """
    A: list = [[0 for j in range (n, len(mtx[i]))]for i in range(n,len(mtx))]
    for i in range(len(mtx)):
        for j in range(len(mtx[i])):
            if n+i < len(mtx[i]):
                if n+j < len(mtx[i]):
                    A[i][j] = mtx[n+i][n+j]
    v: list = reflect_vector(A[0])
    f: list = f_builder(v)
    Q: list = identity(len(mtx))
    for i in range(n,len(Q)):
        for j in range(n, len(Q)):
            Q[i][j] = f[i-n][j-n]
    return Q








def Householder(matrix_A: list) ->list:
    "This function takes a matrix as a lists of list and returns QR factorization using householder"
    """ARGS:
        matrix_A: the input matrix stores as a list of a list
    Returns:
        Q,R: returns the QR factorization of the matrix_A using the Householder algorithm."""
    R: list = deep_copy(matrix_A)
    Q_list: list = []
    for index in range(len(R)):
        Q_temp: list = Q_build(R,index)
        R = LA.matrix_matrix_multi(Q_temp, R)
        Q_list.append(Q_temp)
    Q: list = Q_list[-1]
    Q = conjugate_transpose(Q_list[0])
    for index1 in range(1, len(Q_list)):
        ct: list = conjugate_transpose(Q_list[index1])
        Q = LA.matrix_matrix_multi(Q, ct)
    return Q, R





