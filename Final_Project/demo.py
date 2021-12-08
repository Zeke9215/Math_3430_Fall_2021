import LA
import QR
import LS


scalar_01 = 2

test_vector_01 =[1,2]
test_vector_02 = [3,3]
test_vector_03 = [5,4]

test_matrix_01 = [[3,2],[1,2]]
test_matrix_02 = [[5,5],[5,9]]
test_matrix_03 = [[7,8],[8,2]]

test_matrix_04 = [[1,2,3],[2,2,2],[8,8,8]]
test_matrix_05 = [[3,3,3],[1,1,1],[5,5,5]]


scalar_a = -9
scalar_b = -15
p_norm_vector_01 = [1,2,3]
p_norm_scalar_01 = 2
p_norm_vector_02 = [-1,2,-3]
p_norm_scalar_02 = 3
infinity_vector_01 = [1,2+5j,3]
infinity_vector_02 = [-20,2,3]
boolean_p_norm_vector_01 = [4,2,5]
boolean_p_norm_vector_02 = [3,5,6]
inner_product_vector_01 = [1,2,5+6j]
inner_product_vector_02 = [4,3,5+6j]
inner_product_vector_03 = [1,5,7]
inner_product_vector_04 = [3,5,6+3j]



matrix_qr1 = [[1,1,1],[-1,1,0],[1,2,1]]
matrix_qr2 = [[1,1,3],[2,2,2],[2,2,-1]]

house_matrix1 = [[2,2,1],[-2,1,2],[18,0,0]]
house_matrix2 = [[2,2,1],[-2,1,2],[20,5,0]]


bs_matrix_1 = [[2,1,3],[1,1,2],[2,2,1]]
bs_vector_1 = [4,4,2]

bs_matrix_2 = [[1,2,3],[2,2,1],[2,1,2]]
bs_vector_2 = [2,2,1]


print(LA.add_vectors(test_vector_01,test_vector_02))
print(LA.scalar_vector_multi(test_vector_01,test_vector_02))
print(LA.scalar_matrix_multi(scalar_01,test_matrix_01))
