import numpy as np

def newton_method(f, x0, h=1e-6, eps=1e-6, max_iter=100):
    """Newton's method for solving a scalar nonlinear algebraic equation.

    Args:
        f (function): The nonlinear function.
        x0 (float): The initial guess.
        h (float, optional): The step size for the finite difference approximation of the derivative. Defaults to 1e-6.
        eps (float, optional): The desired accuracy. Defaults to 1e-6.
        max_iter (int, optional): The maximum number of iterations. Defaults to 100.

    Returns:
        float: The solution of the equation.
    """
    x = x0
    for i in range(max_iter):
        fx = f(x)
        dfdx = (f(x + h) - f(x - h)) / (2*h)
        if abs(fx) < eps:
            return x
        x = x - fx/dfdx
    raise ValueError("The method did not converge.")

def f(x):
    return []


# import numpy as np

def sys_lin_eq(L, U, x, P, b):
    N = U.shape[0]
    y = np.zeros((N, 1))
    Pb = P @ b
    count = 10

    for i in range(N):
        y[i, 0] = Pb[i, 0]
        for j in range(i):
            y[i, 0] -= y[j, 0] * L[i, j]
            count += 1
        y[i, 0] /= L[i, i]
        count += 1

    for i in range(N - 1, -1, -1):
        x[i, 0] = y[i, 0]
        for j in range(N - 1, i, -1):
            x[i, 0] -= x[j, 0] * U[i, j]
            count += 1
        x[i, 0] /= U[i, i]
        count += 1
# import numpy as np

def jacobi(x):
    J = np.zeros((10, 10))
    J[0][0] = -x[1][0] * np.sin(x[1][0] * x[0][0])
    J[0][1] = -x[0][0] * np.sin(x[1][0] * x[0][0])
    J[0][2] = 3.0 * np.exp(- (3.0 * x[2][0]))
    J[0][3] = x[4][0] ** 2
    J[0][4] = 2.0 * x[3][0] * x[4][0]
    J[0][5] = -1.0
    J[0][6] = 0.0
    J[0][7] = -2.0 * np.cosh(2.0 * x[7][0]) * x[8][0]
    J[0][8] = -np.sinh(2.0 * x[7][0])
    J[0][9] = 2.0
    J[1][0] = x[1][0] * np.cos(x[1][0] * x[0][0])
    J[1][1] = x[0][0] * np.cos(x[1][0] * x[0][0])
    J[1][2] = x[8][0] * x[6][0]
    J[1][3] = 0.0
    J[1][4] = 6.0 * x[4][0]
    J[1][5] = -np.exp(-x[9][0] + x[5][0]) -  x[7][0] - 1.0
    J[1][6] = x[2][0] * x[8][0]
    J[1][7] = -x[5][0]
    J[1][8] = x[2][0] * x[6][0]
    J[1][9] = np.exp(-x[9][0] + x[5][0])
    J[2,:] = [1,-1,1,-1,1,-1,1,-1,1,-1]
    J[3][0] = - x[4][0] * (x[2][0] + x[0][0])**-2.0
    J[3][1] = -2.0 * x[1][0] * np.cos(x[1][0] * x[1][0])
    J[3][2] = - x[4][0] * (x[2][0] + x[0][0])**-2.0
    J[3][3] = -2.0 * np.sin(-x[8][0] + x[3][0])
    J[3][4] = 1.0 / (x[2][0] + x[0][0])
    J[3][5] = 0
    J[3][6] = -2.0 * np.cos(x[6][0] * x[9][0]) * x[9][0] * np.sin(x[6][0] * x[9][0])
    J[3][7] = -1
    J[3][8] =2.0 * np.sin(-x[8, 0] + x[3, 0])
    J[3][9] = -2.0 * np.cos(x[6, 0] * x[9, 0]) * x[6, 0] * np.sin(x[6, 0] * x[9, 0])
    J[4][0] = 2 * x[7, 0]
    J[4][1] = -2.0 * np.sin(x[1, 0])
    J[4][2] = 2 * x[7, 0]
    J[4][3] = np.power(-x[8, 0] + x[3, 0], -2.0)
    J[4][4] = np.cos(x[4, 0])
    J[4][5] = x[6, 0] * np.exp(-x[6, 0] * (-x[9, 0] + x[5, 0]))
    J[4][6] = -(x[9, 0] - x[5, 0]) * np.exp(-x[6, 0] * (-x[9, 0] + x[5, 0]))
    J[4][7] = 2.0 * np.cosh(2.0 * x[7, 0]) * x[8, 0]
    J[4][8] = 2.0 * np.cosh(2.0 * x[7, 0])
    J[4][9] = x[6, 0] * np.exp(-x[6, 0] * (-x[9, 0] + x[5, 0]))
    J[5][0] = 1.0
    J[5][1] = 1.0
    J[5][2] = -1.0
    J[5][3] = -1.0
    J[5][4] = 1.0
    J[5][5] = -1.0
    J[5][6] = 1.0
    J[5][7] = -1.0
    J[5][8] = 1.0
    J[5][9] = -1.0
    J[6][0] = -np.sinh(x[8, 0] / x[9, 0])
    J[6][1] = x[5, 0] * x[4, 0] * np.exp(x[4, 0] * (x[1, 0] + x[4, 0])) 
    J[6][2] = -1.0
    J[6][3] = -x[2, 0] * np.cos(x[7, 0] * x[3, 0])
    J[6][4] = x[1, 0] * np.exp(x[4, 0] * (x[1, 0] + x[4, 0]))
    J[6][5] = x[4, 0] * np.exp(x[4, 0] * (x[1, 0] + x[4, 0]))
    J[6][6] = -10.0 / x[0, 0]
# J[6][7] = -x[3, 0] * np.cos(x[
    J[6][7] = -np.cos(x[9][0] / x[4][0] + x[7][0])
    J[6][8] = 0
    J[6][9] = -1.0 / x[4][0] * np.cos(x[9][0] / x[4][0] + x[7][0])
    J[7][0] = 2.0 * x[4][0] * (x[0][0] - 2.0 * x[5][0])
    J[7][1] = -x[6][0] * np.exp(x[1][0] * x[6][0] + x[9][0])
    J[7][2] = -2.0 * np.cos(-x[8][0] + x[2][0])
    J[7][3] = 1.5
    J[7][4] = np.power(x[0][0] - 2.0 * x[5][0], 2.0)
    J[7][5] = -4.0 * x[4][0] * (x[0][0] - 2.0 * x[5][0])
    J[7][6] = -x[1][0] * np.exp(x[1][0] * x[6][0] + x[9][0])
    J[7][7] = 0
    J[7][8] = 2.0 * np.cos(-x[8][0] + x[2][0])
    J[7][9] = -np.exp(x[1][0] * x[6][0] + x[9][0])
    J[8][0] = -3
    J[8][1] = -2.0 * x[7][0] * x[9][0] * x[6][0]
    J[8][2] = 0
    J[8][3] = np.exp((x[4][0] + x[3][0]))
    J[8][4] = np.exp((x[4][0] + x[3][0]))
    J[8][5] = -7.0 * np.power(x[5][0], -2.0)
    J[8][6] = -2.0 * x[1][0] * x[7][0] * x[9][0]
    J[8][7] = -2.0 * x[1][0] * x[9][0] * x[6][0]
    J[8][8] = 3
    J[8][9] = -2.0 * x[1][0] * x[7][0] * x[6][0]
    J[9][0] = x[9][0]
    J[9][1] = x[8][0]
    J[9][2] = -x[7][0]
    J[9][3] = np.cos(x[3][0] + x[4][0] + x[5][0]) * x[6][0]
    J[9][4] = np.cos(x[3][0] + x[4][0] + x[5][0]) * x[6][0]
    J[9, 5] = np.cos(x[3, 0] + x[4, 0] + x[5, 0]) * x[6, 0]
    J[9, 6] = np.sin(x[3, 0] + x[4, 0] + x[5, 0])
    J[9, 7] = -x[2, 0]
    J[9, 8] = x[1, 0]
    J[9, 9] = x[0, 0]


def luMatrix(A):
    n=A.shape[0]
    L = np.identity(n)
    U = A.copy().astype(float)
    P = np.identity(n)
    Q = np.identity(n)
    difP,difQ=0,0
    for k in range(n):
        pivot_index = np.argmax(np.abs(U[k:, k:]))
        i = pivot_index//(n-k)+k
        j = pivot_index%(n-k)+k
        if(k!=i):
            difP+=1
        if(k!=j):
            difQ+=1

    # Swap rows and columns in matrices U and P
        U[[k, i], :] = U[[i, k], :]
        U[:, [k, j]] = U[:, [j, k]]
        P[[k, i], :] = P[[i, k], :]
        Q[:, [k, j]] = Q[:, [j, k]]

    # Compute the multipliers and update the matrices L and U
        L[k+1:,k]=U[k+1:,k]/U[k,k]
        U[k+1:,k:]-=np.outer(L[k+1:,k],U[k,k:])
        U[k+1:,k]=L[k+1:,k]
        
        
# Extract the diagonal and upper triangle of U
    L=np.tril(U)
    for i in range(n):
        L[i,i]=1
    U = np.triu(U)
    return P,L,U,Q,difP,difQ


F = numpy.mat([
    math.cos(x2 * x1) - math.exp(-3 * x3) + x4 * x5 ** 2 - x6 - math.sinh(2 * x8) * x9 + 2 * x10 + 2.000433974165385440,
    math.sin(x2 * x1) + x3 * x9 * x7 - math.exp(-x10 + x6) + 3 * x5 ** 2 - x6 * (x8 + 1) + 10.886272036407019994,
    x1 - x2 + x3 - x4 + x5 - x6 + x7 - x8 + x9 - x10 - 3.1361904761904761904,
    2 * math.cos(-x9 + x4) + x5 / (x3 + x1) - math.sin(x2 ** 2) + math.cos(x7 * x10) ** 2 - x8 - 0.1707472705022304757,
    math.sin(x5) + 2 * x8 * (x3 + x1) - math.exp(-x7 * (-x10 + x6)) + 2 * math.cos(x2) - 1.0 / (-x9 + x4) - 0.3685896273101277862,
    math.exp(x1 - x4 - x9) + x5 ** 2 / x8 + math.cos(3 * x10 * x2) / 2 - x6 * x3 + 2.0491086016771875115,
    x2 ** 3 * x7 - math.sin(x10 / x5 + x8) + (x1 - x6) * math.cos(x4) + x3 - 0.7380430076202798014,
    x5 * (x1 - 2 * x6) ** 2 - 2 * math.sin(-x9 + x3) + 0.15e1 * x4 - math.exp(x2 * x7 + x10) + 3.5668321989693809040,
    7 / x6 + math.exp(x5 + x4) - 2 * x2 * x8 * x10 * x7 + 3 * x9 - 3 * x1 - 8.4394734508383257499,
    x10 * x1 + x9 * x2 - x8 * x3 + math.sin(x4 + x5 + x6) * x7 - 0.78238095238095238096])

J = numpy.mat([[-x2 * math.sin(x2 * x1), -x1 * math.sin(x2 * x1), 3 * math.exp(-3 * x3), x5 ** 2, 2 * x4 * x5,
                -1, 0, -2 * math.cosh(2 * x8) * x9, -math.sinh(2 * x8), 2],
               [x2 * math.cos(x2 * x1), x1 * math.cos(x2 * x1), x9 * x7, 0, 6 * x5,
                -math.exp(-x10 + x6) - x8 - 1, x3 * x9, -x6, x3 * x7, math.exp(-x10 + x6)],
               [1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
               [-x5 / (x3 + x1) ** 2, -2 * x2 * math.cos(x2 ** 2), -x5 / (x3 + x1) ** 2, -2 * math.sin(-x9 + x4),
                1.0 / (x3 + x1), 0, -2 * math.cos(x7 * x10) * x10 * math.sin(x7 * x10), -1,
                2 * math.sin(-x9 + x4), -2 * math.cos(x7 * x10) * x7 * math.sin(x7 * x10)],
               [2 * x8, -2 * math.sin(x2), 2 * x8, 1.0 / (-x9 + x4) ** 2, math.cos(x5),
                x7 * math.exp(-x7 * (-x10 + x6)), -(x10 - x6) * math.exp(-x7 * (-x10 + x6)), 2 * x3 + 2 * x1,
                -1.0 / (-x9 + x4) ** 2, -x7 * math.exp(-x7 * (-x10 + x6))],
               [math.exp(x1 - x4 - x9), -1.5 * x10 * math.sin(3 * x10 * x2), -x6,-math.exp(x1 - x4 - x9),
                2 * x5 / x8, -x3, 0, -x5 ** 2 / x8 ** 2, -math.exp(x1 - x4 - x9), -1.5 * x2 * math.sin(3 * x10 * x2)],
               [math.cos(x4), 3 * x2 ** 2 * x7, 1, -(x1 - x6) * math.sin(x4), x10 / x5 ** 2 * math.cos(x10 / x5 + x8),
                -math.cos(x4), x2 ** 3, -math.cos(x10 / x5 + x8), 0, -1.0 / x5 * math.cos(x10 / x5 + x8)],
               [2 * x5 * (x1 - 2 * x6), -x7 * math.exp(x2 * x7 + x10), -2 * math.cos(-x9 + x3), 1.5,
               (x1 - 2 * x6) ** 2, -4 * x5 * (x1 - 2 * x6), -x2 * math.exp(x2 * x7 + x10), 0, 2 * math.cos(-x9 + x3),
                -math.exp(x2 * x7 + x10)],
               [-3, -2 * x8 * x10 * x7, 0, math.exp(x5 + x4), math.exp(x5 + x4),
                -7.0 / x6 ** 2, -2 * x2 * x8 * x10, -2 * x2 * x10 * x7, 3, -2 * x2 * x8 * x7],
               [x10, x9, -x8, math.cos(x4 + x5 + x6) * x7, math.cos(x4 + x5 + x6) * x7,
                math.cos(x4 + x5 + x6) * x7, math.sin(x4 + x5 + x6), -x3, x2, x1]])

