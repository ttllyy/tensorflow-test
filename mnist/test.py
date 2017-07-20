import numpy as np


def testReshape():

    a = np.array([1, 2, 3, 4, 5, 6, 7, 8])

    b = a.reshape([-1, 2])
    c = a.reshape([1, 2, 4])

    d = a.reshape(2, 4)
    print(d)
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    e = x.reshape((1,9))
    print(e)
    x1 = np.array(e[0])
    print(x1)

    x2 = np.reshape(x1,(3,3))
    print(x2)

    x3 = np.reshape(x1, (9, -1))
    print(x3)
    print("--------")
    print(b)
    print(c)

