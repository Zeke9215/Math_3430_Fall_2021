import QR
import LA

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


    Q, R = QR.Householder(matrix_a)
    Q_1 = QR.conjugate transpose(Q)
    Q_2 = LA.matrix_vector_multi(Q_1,vector_a)
    result = backsub(R, Q_2)
    return result
