import cv2
import scipy.sparse as sc
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance

# WORKING
def function_Q1(img):
    img = cv2.resize(img, dsize=(100, 100), interpolation=cv2.INTER_CUBIC)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    plt.imshow(img)
    plt.savefig("out1.png")
    return img

def compute(img, A, i1, j1, i2, j2):
    if (i1 == i2 and j1 == j2):
        return 1
    elif (A[100*i1+j1][i2*100+j2] != 0):
        return A[100*i1+j1][i2*100+j2]
    else:
        red,green,blue = img[i1][j1]
        if (i1 > 95):
            i1 = i1 - 99
        if (j1 > 95):
            j1 = j1 - 99
        x = [red, green, blue, i1/100, j1/100]
        red2,green2,blue2 = img[i2][j2]
        if (i2 > 95):
            i2 = i2 - 99
        if (j2 > 95):
            j2 = j2 - 99
        y = [red2, green2, blue2, i2/100, j2/100]
        dist = distance.euclidean(x,y)
        w = -4 * (dist*dist)
        res = np.exp(w)
        if (res < 0.95):
            return 0
        else:
            return res

# todo -> Fix Wrap Around
def function_Q2(img):
    A = sc.csr_matrix((10000, 10000)).toarray()
    for i in range(100):
        for j in range(100):
            ver = i*100 + j
            for a in range(5):
                for b in range(5):
                    if (a == 0 and b == 0):
                        continue
                    if ((a!=0) and (b!=0) and (i+a) < 100 and  (j+b) < 100):
                        A[ver][ver + (100*a + b)] = compute(img,A, i, j, i+a, j+b)
                    if ((a!=0) and (i + a) < 100):
                        A[ver][ver + (100*a)] = compute(img, A, i, j, i+a, j)
                    if ((a!=0) and (i - a) > 0):
                        A[ver][ver - (100 * a)] = compute(img, A, i, j,i-a, j)
                    if ((b!=0) and j+b < 100):
                        A[ver][ver +b] = compute(img, A, i, j, i, j+b)
                    if ((b!=0) and j-b >0):
                        A[ver][ver -b] = compute(img, A, i, j, i, j-b)
                    if ((a!=0) and (b!=0) and (i+a) < 100 and  (j-b) > 0):
                        A[ver][ver + (100*a -b)]  = compute(img,A,i, j, i+a, j-b)
                    if ((a!=0) and (b!=0) and (i-a) > 0   and  (j+b) < 100):
                        A[ver][ver + (-100*a +b)] = compute(img,A,i, j, i-a, j+b)
                    if ((a!=0) and (b!= 0) and (i-a) > 0   and   (j-b)> 0):
                        A[ver][ver - (100*a + b)] = compute(img,A,i, j, i-a, j-b)

    sumdeg = np.count_nonzero(A)
    avgdegree = sumdeg / 10000
    numedges = sumdeg  / 2
    print("AVG Degree: "+str(avgdegree)) # TODO TAKE INTO ACCOUNT WEIGHTSS!!!!!
    print("NUM Edges : "+str(numedges))
    return A

def function_Q3 (mat):
    start = np.random.normal(0,1,(10000,1))
    res = power_method(mat,start,7)
    res = np.reshape(res, (100,100))
    plt.imshow(res)
    plt.show()

def power_method(mat,start, k):
    result = start
    for i in range(k):
        result = np.matmul(mat,result)
    return result


def main():
    img = plt.imread("input/bear.png")


# Work With Question 1
    img = function_Q1(img)
# Work With Question 2 TODO WORK OK + OPTIMIZE (Most Consuming Resources)
    Am = function_Q2(img)
# Work With Question 3
    Laplacian = sc.csgraph.laplacian(Am,normed=True) # Laplacian not setting 1 to L[0][0]
    img2 = function_Q3(2*sc.identity(10000) - Laplacian)






















if __name__ == '__main__':
    main()




