from tiles3 import *




size = 500;
floats = [10.2, 3.3, 50]
wrapwidths = [20, 2, 100]
ints = [1,2,3,4,5]
numtilings = 16

t = tiles(size, numtilings, floats, ints)
# [117, 418, 44, 75, 269, 3, 366, 49, 235, 229, 236, 151, 118, 85, 427, 391]
print(t)

t = tileswrap(size, numtilings, floats, wrapwidths, ints)
print(t)