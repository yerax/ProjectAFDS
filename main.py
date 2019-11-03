import cv2
import scipy.sparse as sc
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from scipy.sparse.linalg import eigsh

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
        x = [red, green, blue, i1/100, j1/100]
        red2,green2,blue2 = img[i2][j2]
        if (i1 < 50 and i2 > 95):
            i2 = i2 - 99
        if (j2 < 50 and j2 > 95):
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
                    elif ((i+a) > 100 and (j+b) > 100): #wrap around
                        A[ver][ver - 9999 + (100*a + b)] = compute(img,A,i,j, i-99+a, j-99+b)

                    if ((a!=0) and (i + a) < 100):
                        A[ver][ver + (100*a)] = compute(img, A, i, j, i+a, j)
                    elif ((i+a) > 100): #wrap around
                        A[ver][ver - 9900 + (100*a)] = compute(img, A, i,j,i-99+a,j)

                    if ((a!=0) and (i - a) > 0):
                        A[ver][ver - (100 * a)] = compute(img, A, i, j,i-a, j)
                    elif ((i-a) < 0):
                        A[ver][ver + 9900 - (100*a)] = compute(img, A, i, j, i+99-a, j)

                    if ((b!=0) and j+b < 100):
                        A[ver][ver +b] = compute(img, A, i, j, i, j+b)
                    elif ((j+b) > 100):
                        A[ver][ver-99+b] = compute(img,A,i,j,i,j-99+b)

                    if ((b!=0) and j-b >0):
                        A[ver][ver -b] = compute(img, A, i, j, i, j-b)
                    elif ((j-b) < 0):
                        A[ver][ver +99-b] = compute(img, A, i,j,i,j+99-b)

                    if ((a!=0) and (b!=0) and (i+a) < 100 and  (j-b) > 0):
                        A[ver][ver + (100*a -b)]  = compute(img,A,i, j, i+a, j-b)
                    elif ((i+a) > 100 and (j-b) < 0):
                        A[ver][ver -9900 +99 +(100*a -b)] = compute(img, A,i,j,i-99+a, j+99-b)

                    if ((a!=0) and (b!=0) and (i-a) > 0   and  (j+b) < 100):
                        A[ver][ver + (-100*a +b)] = compute(img,A,i, j, i-a, j+b)
                    elif ((i-a) < 0 and (j+b) > 100):
                        A[ver][ver +9900 - 99 + (-100*a +b)] = compute(img,A,i,j,i+99-a, j-99+b)

                    if ((a!=0) and (b!= 0) and (i-a) > 0   and   (j-b)> 0):
                        A[ver][ver - (100*a + b)] = compute(img,A,i, j, i-a, j-b)
                    elif ((i-a) < 0 and (j-b)  < 0):
                        A[ver][ver + 9999 - (100*a +b)] = compute(img,A,i,j,i+99-a, j-99+b)


    sumdeg = np.count_nonzero(A)
    avgdegree = sumdeg / 10000
    numedges = sumdeg  / 2
    print("AVG Degree: "+str(avgdegree)) # TODO TAKE INTO ACCOUNT WEIGHTSS!!!!!
    print("NUM Edges : "+str(numedges))
    return A

def function_Q3 (mat):
    start = np.random.normal(0, 1, (10000,1))
    res = power_method(mat,start,10)
    res = np.reshape(res, (100,100))
    plt.imshow(res)
    plt.savefig("out2.png")

def power_method(mat,start, k):
    result = np.array(start)
    for i in range(k):
        result = mat*result
        result = result / np.linalg.norm(result)
    return result


def getLaplacian (Adj):
    D = sc.csr_matrix((10000, 10000)).toarray()
    Dp = sc.csr_matrix((10000, 10000)).toarray()
    for i in range(10000):
        zeros = np.count_nonzero(Adj[i])
        if (zeros > 0):
            D[i][i] = 1 / np.sqrt(zeros)
            Dp[i][i] = np.sqrt(zeros)
        else:
            D[i][i] = 0
    D = sc.dia_matrix(D)
    Dp = sc.dia_matrix(Dp)
    Laplace = sc.identity(10000) - D * Adj * D
    return Laplace, Dp

def compute2(S):
    print("TODO")


def function_Q4(img, Dp):
    x = np.random.normal(0,1,(10000,1))
    f1 = Dp*np.full((10000,1),1)
    f2 = x - f1 * np.vdot(x, f1)
    # Sort Array
    list = np.empty((10000,2))
    for i in range(10000):
        list[i][0] = i
        list[i][1] = f2[i]
    listSorted = list[list[:,1].argsort()]
    # Algorithm 2
    t = 0
    S = []
    Sp = []
    Sp.append(listSorted[0][0])
    while t < 10000:
        t = t + 1
        S.append(listSorted[t][0])
        if compute2(S) < compute2(Sp):
            Sp = S


    newImg = np.full((10000,1), 0)
    for v in range(len(Sp)):
        newImg[listSorted[Sp[v]]] = 1






def main():
    img = plt.imread("input/bear.png")
# Work With Question 1
    img = function_Q1(img)
# Work With Question 2 TODO WORK OK + OPTIMIZE (Most Consuming Resources)
    Am = function_Q2(img)
# Work With Question 3
    Laplacian,D = getLaplacian(Am)
    img2 = function_Q3(Laplacian)
    #img2 = function_Q3(2*sc.identity(10000) - Laplacian)

# Work With Question 4
   # function_Q4(img2, D)






















if __name__ == '__main__':
    main()




