"""
For this homework assignment we will take our work from HW01 and use it to
prepare a python script which will implement our algoirthms as python functions.

For Problems #0-5 from HW01, Do the following.



1) Write your answer from HW01 in a comment.

2) Below the comment write a function which implements the algorithm from your
comment. If you find that you need to change your algorithm for your python
code, you must edit your answer in the comment.

3) Test each of your functions on at least 2 inputs.

4) Upload your .py file to a github repo named "Math_3430_Fall_2021"

This assignment is due by 11:59pm 09/27/2021. Do NOT upload an updated version to github
after that date.
"""




#Problem 00

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
#Test Inputs
vector_a = test_vector_01 = [1, 2, 4]
vector_b = test_vector_02 = [3, 1, 2]

def add_vectors(vector_a,vector_b):
  result = [0 for element in vector_a]
  for index in range(len(result)):
    result[index] = vector_a[index] + vector_b[index]
  return result

# add_vectors(test_vector_01,test_vector_02) should output [4,3,6]
print(add_vectors(vector_a,vector_b))
print('should be [4,3,6]')
#End Problem0

#Problem1 Scalar-Vector Multiplication
"""
-The Three Questions

Q1: What do we have?
A1: A vector stored as a list. A scalar to multiply each vector component.
Q2: What do we want?
A2: The stored vector multiplied by the scalar. Each component multipled by the scalar and returned as a new vector. 
Q3: How will we get there?
A3: store the vector components as a list. 
mulitply each component by the scalar 
return new vector with components multiplied by the scalar. 

-PsuedoCode

Define vector_scalar_multi(vector_a,scalar_a)
initialize result as an empty list of apropriate size to store the products of the components times the scalar.
Add an element to result for each element in vector_a. set to zero
#set each element of result to be equal to the component times the scalar 
result = vector_a * scalar
return result

vector_a =(a1,a2,..an) #result vector same size as vector_a stored with zeros until replaced with vector components * scalar 
Scalar_a= (c1,c1,..c1n) #same length as vector_a
result[]
vector_a * scalar_a = (a1,a2,..an) * (c1,c1,c1..) = (c1a1,c1,a2,..c1an) = result
return result



Return the desired result.
"""
#1b
"Scalar Vector multiplication"

#Test Inputs
vector_a = test_vector_01 = [1, 2, 4]
vector_b = test_vector_02 = [3, 1, 2]
scalar_b = 2

def scalar_vector_multi(vector_a,scalar_b):
    result = [vector_a]
    result = [index * scalar_b for index in vector_a]
    return result

def scalar_vector_multi(vector_b,scalar_b):
    result = [vector_b]
    result = [index * scalar_b for index in vector_b]
    return result


#Test Inputs

vector_a = test_vector_01 = [1, 2, 4]
vector_b = test_vector_02 = [3, 1, 2]
scalar_b = 2


print(scalar_vector_multi(vector_a,scalar_b))
print('should be[2,4,8]')
print(scalar_vector_multi(vector_b,scalar_b))
print('should be[6,2,4]')

#End Problem 1

#Problem2 Scalar-Matrix Multiplication

"""
-The Three Questions

Q1: What do we have?
A1: matrix as a list of stored columns. A scalar to multiply each element.
Q2: What do we want?
A2: Multiply each list of columns by the scalar and return new result.
Q3: How will we get there?
A3: Multiply each column list by the scalar. return new result of each element multiplied by scalar.
Similar to problem 1 except the vector is a matrix and stored as a List of column vectors.  

-PsuedoCode

Define list of matrix_a
matrix_a = [column_1, column_2, column_3...column_N] #matrix
column_1 = [x11, x21, x31... xn1] #components
column_2 = [x12, x22, x32....xn2]
Column_3 = [x13, x23, x33....xn3]
column_N = [xn1, xn2, xn3....xnn]   
scalar_a = [c1,c1,c1...] #constant and same size as column vectors

Define matrix_scalar_multi(matrix_a,scalar_a)
result [] of zeros equal to length of matrix_a
multiply each column vector by scalar_a
column_1 * scalar_a = [x11,x21,x31,..xn] * [c1,c1,..c1] = [c1x11,c1x21,c1x31..c1xn1] = result
return new column_1 and place in matrix element column_1
continue for all column vectors
return matrix_a with scalar multiplied column vectors instead of zeros
return result matrix_a
end

"""
#Test Inputs

matrix_a = test_matrix_02 = [[1, 1, 1],[9,8,7],[3,7,7]]
vector_a = [1,1,1]
vector_b = [9,8,7]
vector_c = [3,7,7]
scalar_b = 2
matrix_b = test_matrix_01 = [[1, 2, 4],[3,1,2],[5,7,5]]
vector_d = [1,2,4]
vector_e = [3,1,2]
vector_f = [5,7,5]



def scalar_matrix_multi(scalar_b,matrix_a):
    result = [matrix_a]
    result = [index * scalar_b for index in vector_a],[index * scalar_b for index in vector_b],[index * scalar_b for index in vector_c]
    return result

print(scalar_matrix_multi(scalar_b,matrix_a))
print('should be [2,2,2],[18,16,14],[6,14,14]')

def scalar_matrix_multi(scalar_b,matrix_a):
    result = [matrix_b]
    result = [index * scalar_b for index in vector_d],[index * scalar_b for index in vector_e],[index * scalar_b for index in vector_f]
    return result
print(scalar_matrix_multi(scalar_b,matrix_b))
print('should be [2,4,8],[6,2,4],[10,14,10]')

#End Problem2


#Problem3 Matrix-Addition
"""
-The Three Questions
Q1: What do we have?
A1: Two vectors stored as lists. Denoted by the names vector_a and vector_b. 
Q2: What do we want?
A2: Adding corresponding elements of both matrices and return result.
Q3: How will we get there?
A3: Store both matrices as lists of column vectors, and add corresponding elements of corresponding elements of both vectors
return new matrix with the corresponding elements added.

-PsuedoCode
matrix_a = [acolumn1, acolumn2,...acolumn n]
matrix_b = [bcolumn1, bcolumn2,...bcolumn n]

acolumn_1 = [x11, x21, x31... xn1] #components
acolumn_2 = [x12, x22, x32....xn2]
aColumn_3 = [x13, x23, x33....xn3]
acolumn_N = [xn1, xn2, xn3....xnn] 

bcolumn_1 = [x11, x21, x31... xn1] #components
bcolumn_2 = [x12, x22, x32....xn2]
bColumn_3 = [x13, x23, x33....xn3]
bcolumn_N = [xn1, xn2, xn3....xnn] 

define matrix_matrix_add(matrix_a,matrix_b)
result[] #zero vector with the same length as the the matrices to be added.
add corresponding column vectors 
acolumn_1 = [x11, x21, x31... xn1] + bcolumn_1 = [x11, x21, x31... xn1] = result
store added columns in the the first zero placeholder of the new matrix       #result
repeat for each column vector
return added column vectors as a the result
return resulting new matrix       #result = [(acolumn_1+bcolumn_1), (acolumn_2+bcolumn_2)...] 
end

"""
#Test vectors
matrix_a =[[1,1,1],[2,2,2],[3,3,3]]
vector_a = [1,1,1]
vector_b = [2,2,2]
vector_c = [3,3,3]
matrix_b = [[3,3,3],[5,5,5],[1,1,1]]
vector_d = [3,3,3]
vector_e = [5,5,5]
vector_f = [1,1,1]
def add_vectors(vector_a,vector_b):
  result = [0 for element in matrix_a]
  for index in range(len(result)):
    result[index] = vector_a[index] + vector_b[index]
  return result


def matrix_addition(matrix_a,matrix_b):
    result = [matrix_a]
    result = [[add_vectors(vector_a,vector_d)],[add_vectors(vector_b,vector_e)],[add_vectors(vector_c,vector_f)]]
    return result

print(matrix_addition(matrix_a,matrix_b))
print('should be [4,4,4],[7,7,7],[4,4,4]')

#second test
matrix_a =[[3,3,3],[2,2,2],[5,5,5]]
vector_a = [3,3,3]
vector_b = [2,2,2]
vector_c = [5,5,5]
matrix_b = [[1,2,3],[5,7,5],[5,5,5]]
vector_d = [1,2,3]
vector_e = [5,7,5]
vector_f = [5,5,5]
def add_vectors(vector_a,vector_b):
  result = [0 for element in matrix_a]
  for index in range(len(result)):
    result[index] = vector_a[index] + vector_b[index]
  return result


def matrix_addition(matrix_a,matrix_b):
    result = [matrix_a]
    result = [[add_vectors(vector_a,vector_d)],[add_vectors(vector_b,vector_e)],[add_vectors(vector_c,vector_f)]]
    return result

print(matrix_addition(matrix_a,matrix_b))
print('should be [4,5,6],[7,9,7],[10,10,10]')

#EndProblem3


#Problem4:

"""
-The Three Questions
Q1: What do we have?
A1: A matrix stored as a list of column vectors. a vector as a list of elements
Q2: What do we want?
A2: the product of the column vectors with the corresponding vector element. Add the resulting columns and store as new vector.
since matrix vector multiplication results in a vector of the product of the columns with the corresponding vector element, then summed. to give resulting vector.
q3. How do we get there? 
A3. represent the matrix as a list of column vectors. multiply each column vector by the corresponding element of the vector of elements.

-PsuedoCode

# Ax = x1[n11,n21..] + x2[n12,n22] + ...
multiply column 1 by x1, column2 by x2, column 3 by x3 and so on. then sum the column vectors.  

def matrix_vector_multi(matrix_a,vector_a)
matrix_a = [column1, column2,...column n] #list of column vectors
vector_a = [c1,c2,c3,..cn]
resultFinal [] #zero vector as placeholder of same length as vector. #a summed column vector of (1xn)

#scalar-vector multiplication from problem 1
multiply column1 by scalar c1.
column_1 = [x11, x21, x31... xn1]
 Scalar_a= (c1,c1,..c1n) #same length as column_1
result[]
column_1 * scalar_a = (x11,x12,..an) * (c1,c1,c1..) = (c1x11,c1x21,c1x31,..c1an1) = result1
result1

#repeat for every column
multiply column2 by scalar c2.
column_2 = [x12, x22, x32... xn2]
 Scalar_b= (c2,c2,..c2n) #same length as column_2
result[]
column_2 * scalar_b = (x12,x22,..an2) * (c2,c2,c2..) = (c2x12,c2x22,c2x32,..c2an3) = result2
esult2

#scalar_a, scalar_b,..scalar_n are the scalar elements of vector_a
After scalar-Vector multiplication of all column vectors by their respective scalar, add the column vectors.

#From vector vector addition problem 0
def add_vectors(column_1,column_2,column_3...,column_n):

# Initializing result as an empty list
result = []

# Add an element to result for each element of vector_a. Set that element to 0.
for element in vector_a:
  append 0 to result

# Set each element of result to be equal to the desired sum.
for index in range(length(result)):
  result[index] = column_1[index] + column_2[index] + column_3[index] ....column_n[index]

# Return the desired result.
return resultFinal         # first zero vector as placeholder for final answer.
end.


"""
"Matrix-Vector Multiplication"

matrix_a = [[1,1,1],[2,2,2],[3,3,3]]
vector_a = [1,1,1]
vector_b = [2,2,2]
vector_c = [3,3,3]
vector_x = [4,5,7]
scalar_a = 4
scalar_b = 5
scalar_c = 7

column_vector_a = [4,4,4]
column_vector_b = [10,10,10]
column_vector_c = [21,21,21]

def scalar_vector_multi_a(vector_a,scalar_a):
    result = [vector_a]
    result = [index * scalar_a for index in vector_a]
    return result

def scalar_vector_multi_b(vector_b,scalar_b):
    result = [vector_b]
    result = [index * scalar_b for index in vector_b]
    return result

def scalar_vector_multi_c(vector_c,scalar_c):
    result = [vector_c]
    result = [index * scalar_c for index in vector_c]
    return result

def vector_matrix_multi(matrix_a,vector_x):
    result = [0 for elements in vector_x]
    result = [[scalar_vector_multi_a(vector_a,scalar_a)],[scalar_vector_multi_b(vector_b,scalar_b)],[scalar_vector_multi_c(vector_c,scalar_c)]]
    return result

def add_vectors(column_vector_a,column_vector_b):
  result = [column_vector_a,column_vector_b,column_vector_c]
  for index in range(len(result)):
    result[index] = column_vector_a[index] + column_vector_b[index] + column_vector_c[index]
  return result



print(add_vectors(column_vector_a,column_vector_b))
print('should be [35,35,35]')

#End Problem4

#Problem5

"""
-The Three Questions

Q1: What do we have?

A1:  Two matrix stored as column vectors.

Q2: What do we want?

A2: the product of both matrices

Q3: How will we get there?

A3: Very similar to problem 4. Except the vector(nx1) is a matrix (nxm) meaning 
that the column vectors will be multiplied by the multiple vectors that make up a matrix.

-PsuedoCode

#from problem 4
Do the same algorith of scalar vector multiplication except that the column vectors will be multiplied by multiple scalars that make up a row instead of just one of the vector.

# Ax = [x11[n11,n21..]+[x12[n11,n21..]..]
#ColumnVectors*Rowvectors = column_vector1 * row_vector1= (column_vector1*x11) + (column_vector1*x12)...(column_vector1*x1n)
#multipliy each column vector by each scalar of the corresponsing row.

def_matrix_matrix_multi(matrix_a,matrix_b)
matrix_a = [column1, column2,...column n] #list of column vectors
matrix_b = [row1,row2,row3,..row n]
resultFinal [] #zero vector as placeholder of same length as matrix

multiply column1 by scalar a11.
column_1 = [x11, x21, x31... xn1]
 Scalar_a11= (c1,c1,..c1n) #same length as column_1
result[]
column_1 * scalar_a11 = (x11,x12,..an) * (c1,c1,c1..) = (c1x11,c1x21,c1x31,..c1an1) = result1
result1

multiply column1 by scalar a21.
column_1 = [x11, x21, x31... xn1]
 Scalar_a21= (c2,c2,..c2n) #same length as column_1
result[]
column_1 * scalar_a21 = (x11,x12,..an) * (c2,c2,c2..) = (c2x11,c2x21,c2x31,..c2an1) = result2
result2

#From vector vector addition problem 0
def add_vectors(column_1,column_2,column_3...,column_n):

# Initializing result as an empty list
result = []

# Add an element to result for each element of vector_a. Set that element to 0.
for element in vector_a:
  append 0 to result

# Set each element of result to be equal to the desired sum.
for index in range(length(result)):
  result[index] = column_1[index] + column_2[index] + column_3[index] ....column_n[index]

#this sums the column vectors multiplied by the respective row scalars
# Return the desired result.
return resultFinal1         # first zero vector as placeholder for final answer.
                            # in matrix multiplication multiple resultfinals to store in final matrix 
end.



#scalar a11 and a21 is the multiplication of the column vectors with their respective row scalar.
Multiple column with all scalars in the corresponding row, then add the new column vector
Then proceed to the second column vector and corresponding second row(multiplying the column by the scalars in the row), then add
the new second column vectors.

repeat algorith for all column vectors.
multiply column vectors by respective row scalars, then add the new column vectors.
store as resultFinal [1,2,3,4...n] which is the list of the column vectors multiplied by the row scalar then summed
list of new column vectors results in the new matrix.

"""
#### I could not figure out how to relate problem 4 and 5. I know there is a way to simply multiply CORRESPONDING column vectors
### and then sum the CORRESPONNDING elements of those vectors i just could not keep track. Here is my best shot.

"Matrix-matrix Multiplication"



#End

