#Ezekiel Anguiano
#math3430
#edited 3 questions and psuedocode original is on previous submission on github



#0 Vector Addition
"""
-The Three Questions
Q1: What do we have?
A1: Two Vectors stored in a computer named vector_a and vector_b
Q2: What do we want?
A2: Their sum stored as a list.
Q3: How will we get there?
A3: We will create an empty list of the appropriate size and store the sums of
the corresponding components of vector_a and vector_b.
-PsuedoCode
def add_vectors(vector_a,vector_b):
Initialize a result vector of 0's which is the same size as vector_a. Call this
vector result.
# Set each element of result to be equal to the desired sum.
for index in range(length(result)):
  result[index] = vector_a[index] + vector_b[index]
Return the desired result.
"""

vector_a = [8,9,10]
vector_b = [3,3,3]

vector_y = [7,7,7]
vector_z = [3,3,3]

def add_vectors(vector_a,vector_b):
  result = [0 for element in vector_a]
  for index in range(len(result)):
    result[index] = vector_a[index] + vector_b[index]
  return result


def add_vectors(vector_y,vector_z):
  result = [0 for element in vector_y]
  for index in range(len(result)):
    result[index] = vector_y[index] + vector_z[index]
  return result



print("problem#0")
print(add_vectors(vector_a, vector_b))
print("answer for Q0 should be [11,12,13]")
print("test input#2 for problem0")
print(add_vectors(vector_y,vector_z))
print("the answer should be[10,10,10]")



#1 Scalar-Vector Multiplication
"""
-The Three Questions
Q1: What do we have?
A1: A list stored as a Vector, and a Scalar
Q2: What do we want?
A2: The elements in the list to be multiplied by the scalar and returned as a new list. 
Q3: How will we get there?
A3: set the list of size vector_a to a vector of zeros. multiple the elements of the vector 
by the scalar and override each zero element with the appropriate scalar multiplied element

-PsuedoCode
def scalar_vector_multi(scalar,vector):
Initialize a result vector of 0's which is the same size as the vector. Call this
vector result.
# Set each element of result to be multiplied by the scalar
for index in range(length(result)):
  result[index] = scalar * vector[index]
Return the desired result.
"""




vector_a = [1,2,3]
scalar_a = 3

vector_q = [3,3,3]
scalar_q = 10

def scalar_vector_multi(scalar_a,vector_a):
    result = [0 for element in vector_a]
    for index in range(len(vector_a)):
        result[index] = scalar_a * vector_a[index]
    return result

def scalar_vector_multi(scalar_q,vector_q):
    result = [0 for element in vector_q]
    for index in range(len(vector_q)):
        result[index] = scalar_q * vector_q[index]
    return result



print("problem#1")
print(scalar_vector_multi(scalar_a,vector_a))
print("answer for Q1 should be [3,6,9]")
print("test input#2 for problem1")
print(scalar_vector_multi(scalar_q,vector_q))
print("the answer should be [30,30,30]")

#2 Scalar-Matrix Multiplication
"""
-The Three Questions
Q1: What do we have?
A1: A scalar and a matrix stored in a computer. Matrix as a list of a list.
Q2: What do we want?
A2: We want each element in each list multiplied by a scalar.
Q3: How will we get there?
A3: We will create a list of a list and multiply each list component of the matrix by the scalar
and return a new matrix multiplied by the scalar.
-PsuedoCode
def scalar_matrix_multi(scalar,matrix):
Initialize a result matrix as a matrix of zeros. call this zero matrix result.
call function scalar_vector_multi to multiply the lists in the matrix by the scalar
for index in range(length(matrix)):
  result[index] = scalar_vector_multi(scalar,matrix[index]
Return the desired result.
"""


matrix_a = [[2,1,2,],[3,1,3],[5,5,5]]
scalar_b = 5

matrix_w = [[1,2,3],[3,2,1],[5,5,5]]
scalar_w = 2


def scalar_matrix_multi(scalar_b,matrix_a):
    result = [0 for elements in matrix_a]
    for index in range(len(matrix_a)):
        result[index] = scalar_vector_multi(scalar_b,matrix_a[index])
    return result


def scalar_matrix_multi(scalar_w,matrix_w):
    result = [0 for elements in matrix_w]
    for index in range(len(matrix_w)):
        result[index] = scalar_vector_multi(scalar_w,matrix_w[index])
    return result

print("problem#2")
print((scalar_matrix_multi(scalar_b,matrix_a)))
print("answer should be [[10,5,10],[15,5,15],[25,25,25] ")
print("test input #2 for problem2")
print(scalar_matrix_multi(scalar_w,matrix_w))
print("the answer should be [[2,4,6],[6,4,1],[10,10,10]")


#3 Matrix Addition
"""
-The Three Questions
Q1: What do we have?
A1: Two matrices stored in a computer 
Q2: What do we want?
A2: the sum of both matrices 
Q3: How will we get there?
A3: We will create an empty list of the appropriate size and store the sums of
the corresponding elements in the matrices. call the function add_vectors to add the corresponding lists 
in the matrices and return a new matrix that is the sum of the two matrix.  
 
-PsuedoCode
def matrix_matrix_add(matrix_b,matrix_c):
Initialize a result matrix of zeros call this result.
add the corresponding list and return a new matrix of added lists
for index in range(length(matrix_b)):
  result[index] = add_vector(matrix_b[index],matrix_c[index]
Return the desired result.
"""



matrix_b = [[1,1,1],[2,2,2],[3,3,3]]
matrix_c = [[10,1,10],[1,1,1],[7,7,7]]

matrix_u = [[6,7,8],[4,4,4],[3,3,3]]
matrix_v = [[2,2,2],[1,0,0],[0,0,1]]

def matrix_matrix_add(matrix_b,matrix_c):
    result = [0 for elements in matrix_b]
    for index in range(len(matrix_b)):
        result[index] = add_vectors(matrix_b[index],matrix_c[index])
    return result


def matrix_matrix_add(matrix_u,matrix_v):
    result = [0 for elements in matrix_u]
    for index in range(len(matrix_u)):
        result[index] = add_vectors(matrix_u[index],matrix_v[index])
    return result

print("Problem#3")
print(matrix_matrix_add(matrix_b,matrix_c))
print("answer should be [[11,11,11],[3,3,3],[10,10,10]]")

print("test input#2 for problem 3")
print(matrix_matrix_add(matrix_u,matrix_v))
print("answer should be [[8,9,10],[5,4,4],[3,3,4] ")





#4 Matrix-Vector multiplication
"""
-The Three Questions
Q1: What do we have?
A1: A matrix as a list of columns, and a vector to be multiplied. 
Q2: What do we want?
A2: the product of the list of columns and the vector.
Q3: How will we get there?
A3: We will multiply the list of columns by the corresponding vector element, then add the resulting lists 
by their corresponding elements to return a new vector.
-PsuedoCode
def matrix_vector_multi(matrix,vector):
Initialize a result matrix of zeros call it result.
for index in range(length(matrix)):
Call scalar_vector_multi and multiply list of column vectors with the corresponding vector component
  result[index] = [scalar_vector_multi(matrix[0],vector[0].... for every corresponding vector component 
call function add_vectors to add the resulting vectors. add the first two then the third 
Return the desired result as a new vector

"""




matrix_d =[[10,10,10],[7,7,7],[15,13,15]]
vector_c = [7,8,9]

matrix_m =[[1,1,1],[9,8,7],[3,4,3]]
vector_n = [6,7,8]

def matrix_vector_multi(matrix_d,vector_c):
    result = [0 for elements in matrix_d]
    for index in range(len(matrix_d)):
        result = [scalar_vector_multi(matrix_d[0],vector_c[0]), scalar_vector_multi(matrix_d[1],vector_c[1]),scalar_vector_multi(matrix_d[2],vector_c[2])]
        result = add_vectors(result[0], result[1])
        result = add_vectors(result, scalar_vector_multi(matrix_d[2],vector_c[2]))

        return result

def matrix_vector_multi(matrix_m,vector_n):
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



#5 Matrix Matrix Multiplication
"""
-The Three Questions
Q1: What do we have?
A1: Two matrices stored as lists of column vectors
Q2: What do we want?
A2: the product of two matrices using their column vectors.
Q3: How will we get there?
A3: multiply the corresponding column vectors of the two matrices and use the function
matrix_vector_multi to compute the result. 
-PsuedoCode
def matrix_matrix_multi(matrix_f,matrix_e):
set result = to matrix_f and overwrite the elements with the result of the function
for index in range(length(matrix_f)):
  call matrix_vector_multiplication and multiply the column vectors of one matrix with the corresponding components of the other matrix.
  result[index] = matrix_vector_multi(matrix_f(0),matrix_e... for all the column vectors and their corresponding components
Return the desired result.
"""


matrix_e = [[3,5,3],[4,8,4],[1,4,1]]
matrix_f = [[2,1,5],[9,2,1],[1,2,7]]

matrix_g = [[1,1,1],[9,9,7],[2,2,1]]
matrix_h = [[0,3,3],[2,3,4],[5,5,5]]

def matrix_matrix_multi(matrix_f,matrix_e):
    result = [matrix_f]
    for index in range(len(matrix_f)):
        result[index]  = matrix_vector_multi(matrix_f[0],matrix_e), matrix_vector_multi(matrix_f[1],matrix_e), matrix_vector_multi(matrix_f[2],matrix_e)
        return result

def matrix_matrix_multi(matrix_h,matrix_g):
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