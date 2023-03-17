from math import sqrt
import numpy as np

eps = 10e-14

def luColumn(A):
    n = A.shape[0]
    L = np.eye(n)
    U = np.copy(A).astype(float)
    P = np.eye(n)
    dif=0
    for k in range(n):
        # Find the index of the pivot element in column k
        pivot_index = np.argmax(np.abs(U[k:, k])) + k
        if(pivot_index!=k):
            dif+=1

        # Swap rows k and pivot_index
        U[[k, pivot_index], :] = U[[pivot_index, k], :]
        P[[k, pivot_index], :] = P[[pivot_index, k], :]
        
        L[[k, pivot_index], :k] = L[[pivot_index, k], :k]

        # Update the L matrix and U matrix
        if(abs(U[k,k])<eps*norm(A)):
            continue
        L[k+1:,k]=U[k+1:,k]/U[k,k]
        U[k+1:,k:]-=np.outer(L[k+1:,k],U[k,k:])

    return P, L, U, dif
def det(A):
    n = A.shape[0]
    P, L, U,dif = luColumn(A)

    det_U = np.prod(np.diag(U))
    det_P = (-1)**dif
    det_L = np.prod(np.diag(L))

    det_A = det_P * det_L * det_U

    return det_A
def luSolve(A, b):

    n = A.shape[0]
    P,L,U,_ = luColumn(A)

    #Ax=b => PAx=Pb => LUx=Pb => Ly=Pb(y=Ux)
    # Solve Ly=b for y

    Pb = np.dot(P, b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = Pb[i] - np.dot(L[i,:i], y[:i])
        #y[i] = Pb[i] - sum(L[i][j]*y[j] for j in range(i))

    # Solve Ux=y for x
    x = np.zeros((n,1))
    for i in range(n-1, -1, -1):
        if(abs(U[i,i])<eps*norm(A)):
            return None
        x[i] = (y[i] - np.dot(U[i,i+1:], x[i+1:])) / U[i,i]
        #x[i] = (y[i] - sum(U[i][j]*x[j] for j in range(i+1, n))) / U[i][i]
    return x
def solveAccuracy(A,b):
    dif=A@luSolve(A,b)-b
    return dif
def luInverse(A):
    n = A.shape[0]
    P,L, U,dif = luColumn(A)

    # Solve AX[:,i] = I[:,i] for each column i of the identity matrix
    # LUX=PAX=PI=> Сначала решаем Ly=PI => решаем UX=y
    X = np.zeros((n, n))
    for i in range(n):
        b = np.zeros(n)
        b[i] = 1.0
        Pb = np.dot(P, b)
        y = np.zeros(n)
        for j in range(n):
            y[j] = Pb[j] - np.dot(L[j,:j], y[:j])
        x = np.zeros(n)
        for j in range(n-1, -1, -1):
            x[j] = (y[j] - np.dot(U[j,j+1:], x[j+1:])) / U[j,j]
        X[:,i] = x
    return X
def inverseAccuracy(A):
    print(A@luInverse(A),"\t: A*A^(-1)")
    print(luInverse(A)@A,"\t: A^(-1)*A")
def norm(A):
    n=A.shape[0]#поменять норму на бесконечность
    result=0
    for i in range(n):
        for j in range(n):
            result+=A[i][j]**2
    result=sqrt(result)
    return result

def luCond(A):
    return norm(A)*norm(luInverse(A))

def rank(A):
    U=A.copy().astype(float)
    n=A.shape[0]
    resultRank = n
    i=0
    while i <= n-1:
        if(i>=resultRank):
            return resultRank
        j = np.argmax(np.abs(U[i:,i])) + i
        if(abs(U[j][i])<eps*norm(A)):
            U[:,[i,resultRank-1]]=U[:,[resultRank-1,i]]
            resultRank-=1
            i-=1
        else:
            U[[i,j],:]=U[[j,i],:]
            U[i+1:,i:resultRank]-=np.outer((U[i+1:,i]/U[i,i]),U[i,i:resultRank])
        i+=1
    return resultRank
            
def luSolve2(A,b):
    P,L,U,dif = luColumn(A)
    detA=det(A)
    if(abs(det(A))>eps*norm(A)):
        return luSolve(A,b)
    
    #Ly=Pb
    n=A.shape[0]
    Pb=np.dot(P,b)
    y=np.zeros((n,1))
    for i in range(n):
        a=Pb[i]
        v=np.dot(L[i,:i],y[:i])
        y[i]=Pb[i]-np.dot(L[i,:i],y[:i])
    #y=Ux
    x=np.zeros((n,1))
    numOfVariables=n
    for i in range(n-1,-1,-1):
        if(abs(U[i][i])<eps*norm(U)):
            if(abs(y[i])>eps*norm(U)):
                return None
            else:
                numOfVariables-=1
        else:
            tmp = np.dot(U[i,i+1:(numOfVariables)],x[i+1:(numOfVariables)])
            x[i]=(y[i]-np.dot(U[i,i+1:],x[i+1:]))/U[i][i]
    return x

n=5
# A=np.random.randint(10,size=(n,n))
# A = np.array([[-2,5,1,3],[5, 4, 5,18],[1, 3, 2,7],[8,-9,1,5]])
#A = np.array([[1,3,2],[4,6,5],[7,9,8]])

b=np.random.randint(10,size=(n,1))
A=np.random.rand(n,n)
A[:,2]=A[:,1]+A[:,0]
A[:,3]=A[:,1]-A[:,0]

b = np.dot(A, b)
P,L,U,_=luColumn(A)
print("A:\n")
print(A,"\n\n")
print("L:\n")
print(L,'\n\n')
print("U:\n")
print(U,'\n\n')
print("P:\n")
print(P,'\n\n')
print("LU-PA:\n")
print(L@U-P@A,'\n\n')








#2 rank
print("Rank:\t",rank(A),"\n\n")
x=luSolve2(A,b)
print("Solve result: \n")
print(x,"\n\n")

if x is not None:
    print("Ax-b: \n\n")
    print(A@x-b,"\n\n")








# #0. LU column
# P, L, U,dif = luColumn(A)

# print(P,"\t:P\n")
# print(L,"\t:L\n")
# print(U,"\t:U\n")
# print(P@A,"\t:P*A\n")
# print(L@U,"\t:L*U\n")
# print(dif,"\t:dif\n")
# print(norm(dif),"\t:norm\n\n")

# #a. Determinant

# print("Determinant\n")
# print(det(A),"\n\n")

# #b. Solve Ax=b
# print("Solve\n")
# x=luSolve(A,b)
# if (x is None):
#     print("none")
# else:
#     print(x,"\n")
#     print(solveAccuracy(A,b),"\n\n")

# #c. Inv
# print("Inverse\n")
# if(det(A)!=0):
#     print(luInverse(A),"\n")
#     inverseAccuracy(A)
#     print("\n\n")

# #d. Cond
# print("Condition\n")
# print(luCond(A),"\n\n")