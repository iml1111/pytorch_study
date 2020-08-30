import numpy as np
# 테스트용 DTM 4*9
# 인덱스를 통한 각 단어의 빈도수 카운트
A=np.array([
	[0,0,0,1,0,1,1,0,0],
	[0,0,0,1,1,0,1,0,0],
	[0,1,1,0,2,0,0,0,0],
	[1,0,0,0,0,0,0,1,1]])

#해당 DTM을 SVD로 분해
# A = U*S*VT
U, s, VT = np.linalg.svd(A, full_matrices = True)
S = np.zeros((4,9))
S[:4, :4] = np.diag(s)

# 두개의 행렬이 같으면 return
#print(np.allclose(A, np.dot(np.dot(U,S), VT).round(2)))

# 차원 축소
U = U[:,:2] # 4*2 = 문서의 수 * 토픽의 수
S = S[:2, :2]
VT = VT[:2, :] # 2 * 9 토픽의 수 * 단어의 수
A_prime=np.dot(np.dot(U,S), VT)
print(A_prime.round(2))