

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
