import cv2
import scipy.sparse as sc
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from scipy.sparse.linalg import eigsh
import copy

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
        if (i1 < 90 and i2 > 95):
            i2 = i2 - 100
        if (i1 > 90 and i2 < 5):
            i2 = i2 + 100
        if (j1 < 90 and j2 > 95):
            j2 = j2 - 100
        if (j1 > 90 and j2 < 5):
            j2 = j2 + 100
        y = [red2, green2, blue2, i2/100, j2/100]
        dist = distance.euclidean(x,y)
        w = -4 * (dist*dist)
        res = np.exp(w)
        if (res < 0.95):
            return 0
        else:
            return res

def function_Q2(img):
    A = sc.csr_matrix((10000, 10000)).toarray()
    for i in range(100):
        for j in range(100):
            ver = i*100 + j
            for a in range(5):
                if (a == 0):
                    continue

                if ((i+a) < 100 and (j+a) < 100):
                   comp = compute(img,A, i, j, i+a, j+a)
                   A[ver][ver + (100*a + a)] = comp
                   A[ver + (100*a + a)][ver] = comp
                elif ((i+a) > 100 and (j+a) > 100): #wrap around
                   comp = compute(img,A,i,j, i-99+a, j-99+a)
                   A[ver][ver - 9999 + (100*a + a)] = comp
                   A[ver - 9999 + (100*a + a)][ver] = comp

                if ((i + a) < 100):
                    comp = compute(img, A, i, j, i+a, j)
                    A[ver][ver + (100*a)] = comp
                    A[ver + (100*a)][ver] = comp
                elif ((i+a) > 100): #wrap around
                    comp = compute(img, A, i,j,i-99+a,j)
                    A[ver][ver - 9900 + (100*a)] = comp
                    A[ver - 9900 + (100*a)][ver] = comp

                if ((i - a) > 0):
                    comp = compute(img, A, i, j,i-a, j)
                    A[ver][ver - (100 * a)] = comp
                    A[ver - (100 * a)][ver] = comp
                elif ((i-a) < 0):
                    comp = compute(img, A, i, j, i+99-a, j)
                    A[ver][ver + 9900 - (100*a)] = comp
                    A[ver + 9900 - (100*a)][ver] = comp

                if (j+a < 100):
                    comp = compute(img, A, i, j, i, j+a)
                    A[ver][ver +a] = comp
                    A[ver +a][ver] = comp
                elif ((j+a) > 100):
                    comp = compute(img,A,i,j,i,j-99+a)
                    A[ver][ver-99+a] = comp
                    A[ver-99+a][ver] = comp

                if (j-a >0):
                    comp = compute(img, A, i, j, i, j-a)
                    A[ver][ver -a] = comp
                    A[ver-a][ver]  = comp
                elif ((j-a) < 0):
                    comp = compute(img, A, i,j,i,j+99-a)
                    A[ver][ver +99-a] = comp
                    A[ver+99-a][ver]  = comp


                if ((i+a) < 100 and  (j-a) > 0):
                    comp = compute(img,A,i, j, i+a, j-a)
                    A[ver][ver + (100*a -a)]  = comp
                    A[ver + (100*a -a)][ver]  = comp
                elif ((i+a) > 100 and (j-a) < 0):
                    comp = compute(img, A,i,j,i-99+a, j+99-a)
                    A[ver][ver -9900 +99 +(100*a -a)] = comp
                    A[ver -9900 +99 +(100*a -a)][ver] = comp

                if ((i-a) > 0   and  (j+a) < 100):
                    comp = compute(img,A,i, j, i-a, j+a)
                    A[ver][ver + (-100*a +a)] = comp
                    A[ver + (-100*a +a)][ver] = comp
                elif ((i-a) < 0 and (j+a) > 100):
                    comp = compute(img,A,i,j,i+99-a, j-99+a)
                    A[ver][ver +9900 - 99 + (-100*a +a)] = comp
                    A[ver +9900 - 99 + (-100*a +a)][ver] = comp

                if ((i-a) > 0   and   (j-a)> 0):
                    comp = compute(img,A,i, j, i-a, j-a)
                    A[ver][ver - (100*a + a)] = comp
                    A[ver - (100*a + a)][ver] = comp
                elif ((i-a) < 0 and (j-a)  < 0):
                    comp = compute(img,A,i,j,i+99-a, j-99+a)
                    A[ver][ver + 9999 - (100*a +a)] = comp
                    A[ver + 9999 - (100*a +a)][ver] = comp


    sumdeg = np.count_nonzero(A)
    numedges = sumdeg  / 2
    print("AVG Degree: "+str(sum(A[np.nonzero(A)])/10000)) # TODO TAKE INTO ACCOUNT WEIGHTSS!!!!!
    print("NUM Edges : "+str(numedges))
    return A

def function_Q3 (mat,D):
    res = power_method(mat,D,150)
    res = np.reshape(res, (100,100))
    plt.imshow(res)
    plt.savefig("out2.png")
    return res

def power_method(mat,f_1,k):
    x = np.random.normal(0, 1, 10000)
    f_1 = f_1 / np.sqrt((f_1 ** 2).sum())
    x0 = x - np.inner(f_1, x)*f_1
    for i in range(k):
        x0 = mat.dot(x0)
        x0 = np.squeeze(np.asarray(x0))
        x0/=np.linalg.norm(x0)
    return x0

def compute2(S):
    print("TODO")

def function_Q4(img,Adj):
    f2 = img.reshape(10000)
    # Sort Array
    list = np.empty((10000,2))
    for i in range(10000):
        list[i][0] = i
        list[i][1] = f2[i]
    listSorted = list[list[:,1].argsort()]
    # Algorithm 2
    t = 0
    S = []
    compS = 0
    Sp = []
    Sp.append(listSorted[0][0])
    compSp = np.count_nonzero(Adj[int(listSorted[0][0])])
    while t < 9999:
        t = t + 1
        compS += np.count_nonzero(Adj[int(listSorted[t][0])])
        for i in range (len(S)):
            if (Adj[int(S[i])][int(listSorted[t][0])] != 0):
                compS -= 2 # Erase Edge
        S.append(listSorted[t][0])
        if compS / min(len(S), 10000-len(S)) < compSp / min(len(Sp), 10000-len(Sp)):
            compSp = compS
            Sp = copy.copy(S)

    print(Sp)

    newImg = np.full((10000,1), 0)
    for v in range(len(Sp)):
        newImg[int(listSorted[int(Sp[v])][0])] = 1.0

    newImg = np.reshape(newImg, (100,100))
    plt.imshow(newImg)
    plt.savefig("out3.png")

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
    Laplace = sc.identity(10000) - D * Adj * D
    return Laplace, Dp


def main():
    img = plt.imread("input/bear.png")
# Work With Question 1
    img = function_Q1(img)
# Work With Question 2
    Am = function_Q2(img)
# Work With Question 3
    Laplacian, Di= sc.csgraph.laplacian(Am,normed=True,return_diag=True)
    img2 = function_Q3(2*sc.identity(10000) - Laplacian,Di)
# Work With Question 4
    #function_Q4(img2,Am)




if __name__ == '__main__':
    main()








