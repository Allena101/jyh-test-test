# import copy
import itertools
import numpy as np
import copy

# Here you should add your testing array (for now)
# myArray = np.array([[1,2,3],[4,5,6],[7,8,9]])
myArray = np.array([[1,2,3],[0,1,5],[5,6,0]])

# a = np.arange(8).reshape(2,2,2)

print(myArray)

# print(np.diagonal(myArray, offset=0, axis2=1))
# np.diag(arr[:, ::-1]

# print(np.diag(myArray[:, ::-1]))


# print(myArray[:,::-1])


# np.diag(np.rot90(myArray))

# a2 = myArray[][]

# copy first 2 columns into separate matrix
print(myArray[:,0:2])

a2 = myArray[:,0:2]

# numpy.append(arr, values, axis=None)

# Appends the copied 2 first columns to the original array
extendedArray = np.append(myArray, a2, axis=1)
print(extendedArray)
# print(myArray)

print(np.diag(extendedArray))
diag1Mutt = np.diag(extendedArray)
# for i in range(len(diag1Mutt)-1):


# Adds diagonals from the extendedArray into the diagElementList
diagShiftIx = 0
diagElementList = []
for i in range(len(extendedArray)):
    diag1Mutt = np.diag(extendedArray, k=diagShiftIx)
    diagElementList.append(diag1Mutt)
    diagShiftIx += 1
# shift the amount of times as the for loop and save to list

print(f"{diagElementList=}")



# In [47]: np.diag(np.fliplr(array))


# multiplies the diagonals elements cumulatively
diagList = []
product = 1
for diag in diagElementList:
    for i in diag:
        product = product * i
    diagList.append(product)
    product = 1
print(F'New try {diagList=}')







# addTen = np.vectorize(add)





"""
a = numpy.array([[  0.,  1.,  2.,  3.,  4.],
                 [  7.,  8.,  9., 10.,  4.],
                 [ 14., 15., 16., 17.,  4.],
                 [  1., 20., 21., 22., 23.],
                 [ 27., 28.,  1., 20., 29.]])
print numpy.argwhere(a == 4.)
"""


print(len(extendedArray))
print(extendedArray)

print("OJOJOJ")
# np.diag(np.rot90(a3))

# x = np.rot90(a3)
# x = np.rot90(x)
# x = np.fliplr(x)
#
# print(x)


arrayFlip = extendedArray[:]
arrayFlip = np.flip(arrayFlip, axis=0)
print(arrayFlip)
# Now the same process for the flipped array
diagShiftIxFlip = 0
diagElementListFlip = []
for i in range(len(arrayFlip)):
    diag1Mutt = np.diag(arrayFlip,k=diagShiftIxFlip)
    diagElementListFlip.append(diag1Mutt)
    diagShiftIxFlip += 1
# shift the amount of times as the for loop and save to list

# print(f"{diagElementListFlip}")


# diagListFlip = []
# diagCounterFlip = 0
# for ix, i in enumerate(diag1Mutt[:-1]):
#     diagSum = i*diag1Mutt[ix+1]
#     diagCounter += diagSum
# diagList.append(diagCounter)

diagListFlip = []
product = 1
for diag in diagElementListFlip:
    for i in diag:
        product = product * i
    diagListFlip.append(product)
    product = 1

print(F'Flip cum diag {diagListFlip=}')


determinant = sum(diagList) - sum(diagListFlip)
print(F'{determinant=}')


# np.diag(np.fliplr(array))

# First diag should be 5 1 3


# print((-30) * (1/5))


# x = np.transpose(myArray)


# bigMatrix = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
# print(bigMatrix)
# print(np.transpose(bigMatrix))

nMatrix = np.zeros(myArray.shape, dtype=int)
print(nMatrix)







garbage_list = [1,2]
RCR = itertools.repeat( garbage_list,   times=len(myArray))




def get_cycle(array):
    RowColPermutations = []
    rowCol = 0
    for _ in range(array.size):
        RowColPermutations.append(rowCol)
        if RowColPermutations.count(rowCol) == 3:
            rowCol += 1
    return RowColPermutations

# qqq = get_cycle(myArray)
# print('Test Function')
# print(F'{qqq}')


RowColPermutations = get_cycle(myArray)
matrix_coordinate = itertools.cycle(RowColPermutations)
print(F'{matrix_coordinate=}')


MatrixAxis = itertools.cycle([0,1])
matrix_coordinate123 = itertools.cycle(range(len(myArray)))


miniDiag = []
for i in range(myArray.size):
    # print(F'{col=}')
    x = myArray[:]
    # x = myArray.copy()
    x = np.delete(x, next(matrix_coordinate), next(MatrixAxis))
    x = np.delete(x, next(matrix_coordinate123), next(MatrixAxis))
    miniDiag.append(x)

print("Mini diagonals ↓")
print(F'{miniDiag=}')

# try this one with zip
# mini diag
# unsure if a larger matrix would result in bigger mini diags so lets get back to that one when we are finished!

# CF
# Trying to figure out how to revers a sub list in a numpy array
# Goal reverse teh sublists
print('Testing Here')
# miniDiagFlip = miniDiag.copy()
# test = np.zeros(miniDiag.shape)
print(F'{miniDiag=}')
# print(type(miniDiag))
# print(type(myArray))
# print(type(miniDiag[0]))
miniDiagFlip  = copy.deepcopy(miniDiag)

# x = miniDiag
# print(x)
# print(F'{miniDiagFlip}')

for ix, i in enumerate(miniDiagFlip):
    print(i)
    i = np.flip(i, axis=0)
    miniDiagFlip[ix] = i



print("Flipped Mini diag attempt ↓")
print(F'{miniDiagFlip}')



arrayFlip = np.flip(arrayFlip, axis=0)
print(arrayFlip)
# Now the same process for the flipped array
diagShiftIxFlip = 0
diagElementListFlip = []
for i in range(len(arrayFlip)):
    diag1Mutt = np.diag(arrayFlip,k=diagShiftIxFlip)
    diagElementListFlip.append(diag1Mutt)
    diagShiftIxFlip += 1











diagShiftIx = 0
miniDiagElementList = []
for i in miniDiag:
    diag1Mutt = np.diag(i, k=diagShiftIx)
    miniDiagElementList.append(diag1Mutt)
    # if diagShiftIx
    #     diagShiftIx += 1








print("Testing mini diag ↓")
print(F'{miniDiagElementList}')




# backup is just appending to a list and then creating a new array with that list in the shape of the original array




print(myArray)
print("MMMMMMMM")
# print(myArray[ 0:1, :])
# print(myArray[ :, 1:2])

# A = np.delete(A, 1, 0)  # delete second row of A

x = myArray[:]
y = myArray[:]
z = myArray[:]

a = myArray[:]
b = myArray[:]
c = myArray[:]

m = myArray[:]
n = myArray[:]
q = myArray[:]



x = np.delete(x, 0,0)
x = np.delete(x, 0,1)

y = np.delete(y, 0,0)
y = np.delete(y, 1,1)
#
z = np.delete(z, 0,0)
z = np.delete(z, 2,1)


a = np.delete(a, 1,0)
a = np.delete(a, 0,1)

b = np.delete(b, 1,0)
b = np.delete(b, 1,1)

c = np.delete(c, 1,0)
c = np.delete(c, 2,1)

"""
I think the cykle generator for the col needs to be 0 1 2 , 0 1 2 insted of 111 222 etc. 
So you could make itertools.cycle for that

"""



# 1 2
# 5 6




# b = np.delete(y, 0,0)
# b = np.delete(y, 1,1)
# #
# c = np.delete(z, 0,0)
# c = np.delete(z, 2,1)



# print(x = np.delete(myArray, 0,0, axis=1))
# x = np.delete(x, 0,1)

# print(x)
# print(y)
# print(z)
# print("AAA ↓")
# print(a)
# print(b)
# print(c)

# q = np.delete(myArray, 1, axis=1)
# print(print(F'{q=}'))



# 0 5
# 5 0

# 4 7
# 6 9

"""
check that matrix is square
check for determinant being 0
maybe check if its larger than 3x3 (depends on how things scale)
"""

for i in range(15):
    pass
    # print(next(matrix_coordinate))
    # print(next(matrix_coordinate123))

# tempCounter = 1
# for i in miniDiag:
#     print(F'{tempCounter} ↓')
#     print(i)
#     tempCounter += 1


# print(miniDiag)