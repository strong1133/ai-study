# 파이썬의 배열


arr = [1, 2, 3, 4, 5]
print(arr)

n = 5
arr1 = [0] * n
print(arr1)

n = 5
arr2 = [i for i in range(n)]
print(arr2)


n =3
m=5
arr3=[[0]*m for i in range(n)]
print(arr3)
arr4 = [[i*m+j for j in range(m)] for i in range(n)]
print(arr4)