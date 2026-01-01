import numpy as np
ar = [
[-0.02505078  ,0.46567145 , 0.82728827],
[-0.03642717  ,0.59952235,  0.90071845],
[0.06120982 ,0.10846937, 0.07209069],
]

a = np.array(ar[0])
mean = np.array(ar[1])
std = np.array(ar[2])

b = (a-mean)/std
print(b)
print(b*std+mean)
