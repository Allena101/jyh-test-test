import itertools
import numpy as np

ml = [[1,1,0], [2,5,5], [3,0,6]]
# ml = [i for i in range(11)]

# multi = 0
# cum = 0
# for ix, i in enumerate(ml[:-1]):
#     multi += i*ml[ix+1]
#     print(F'{multi=}')
#     cum += multi
#     print(F'{cum=}')


# print(F'{cum=}')

diagList = []
product = 1
for diag in ml:
    for i in diag:
        product = product * i
    diagList.append(product)
    product = 1




print(F'{diagList=}')

tl = [-10,5,3]

print(sum(tl))




Caxis = itertools.cycle([0,1])
myIter = iter(Caxis)

for i in range(10):
    # print(Caxis)
    # print(next(myIter))
    print(next(Caxis))

ml = [1,1,3,3,3,5]
print("KKK")
# ml.append(5) * 3
print(ml.count(3))


bigMatrix = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])

print(bigMatrix)

# h = a.copy() Create a deep copy of the array
# bigMatrix[:]
x = np.zeros(bigMatrix.shape)

# x = bigMatrix.copy()
x[:,0] = bigMatrix[0,:]
x[:,1] = bigMatrix[1,:]
x[:,2] = bigMatrix[2,:]
x[:,3] = bigMatrix[3,:]
print('Original Matrix ↓')
print(bigMatrix)
print('transpose matrix ↓')
print(x)
# print(bigMatrix)
# x[0,0] = bigMatrix[3,3]
# print(x)

# after comma is the column specified


# print(x[:,2])
# print(x[3,:])




# print(bigMatrix[0,:])
# x[:,0] = bigMatrix[1,:]
