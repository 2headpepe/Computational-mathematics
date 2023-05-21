import numpy as np
import time


def function(x):
    f = np.zeros((10, 1))
    f[0][0] = np.cos(x[1][0] * x[0][0]) - np.exp(-(3.0 * x[2][0])) + x[3][0] * x[4][0] * x[4][0] - x[5][0] - np.sinh((2.0 * x[7][0])) * x[8][0] + (2.0 * x[9][0]) + 2.000433974165385440
    f[1][0] = np.sin(x[1][0] * x[0][0]) + x[2][0] * x[8][0] * x[6][0] - np.exp(-x[9][0] + x[5][0]) + 3.0 * x[4][0] * x[4][0] - x[5][0] * (x[7][0] + 1.0) + 10.886272036407019994
    f[2][0] = x[0][0] - x[1][0] + x[2][0] - x[3][0] + x[4][0] - x[5][0] + x[6][0] - x[7][0] + x[8][0] - x[9][0] - 3.1361904761904761904
    f[3][0] = 2.0 * np.cos(-x[8][0] + x[3][0]) + x[4][0] / (x[2][0] + x[0][0]) - np.sin(x[1][0] * x[1][0]) + np.power(np.cos(x[6][0] * x[9][0]), 2.0) - x[7][0] - 0.1707472705022304757
    f[4][0] = np.sin(x[4][0]) + 2.0 * x[7][0] * (x[2][0] + x[0][0]) - np.exp(-x[6][0] * (-x[9][0] + x[5][0])) + 2.0 * np.cos(x[1][0]) - 1.0 / (-x[8][0] + x[3][0]) - 0.3685896273101277862
    f[5][0] = np.exp(x[0][0] - x[3][0] - x[8][0]) + x[4][0] * x[4][0] / x[7][0] + np.cos(3.0 * x[9][0] * x[1][0]) / 2.0 - x[5][0] * x[2][0] + 2.0491086016771875115
    f[6][0] = np.power(x[1][0], 3.0) * x[6][0] - np.sin(x[9][0] / x[4][0] + x[7][0]) + (x[0][0] - x[5][0]) * np.cos(x[3][0]) + x[2][0] - 0.7380430076202798014
    f[7][0] = x[4][0] * np.power(x[0][0] - 2.0 * x[5][0], 2.0) - 2.0 * np.sin(-x[8][0] + x[2][0]) + 1.5 * x[3][0] - np.exp(x[1][0] * x[6][0] + x[9][0]) + 3.5668321989693809040
    f[8][0] = 7.0 / x[5][0] + np.exp(x[4][0] + x[3][0]) - 2.0 * x[1][0] * x[7][0] * x[9][0] * x[6][0] + 3.0 * x[8][0] - 3.0 * x[0][0] - 8.4394734508383257499
    f[9][0] = x[9][0] * x[0][0] + x[8][0] * x[1][0] - (x[7][0] * x[2][0]) + np.sin(x[3][0] + x[4][0] + x[5][0]) * x[6][0] - 0.78238095238095238096
    return f
def jacobi(x):
    J = np.zeros((10, 10))

    J[0, 0] = -x[1][0] * np.sin(x[1][0] * x[0][0])
    J[0, 1] = -x[0][0] * np.sin(x[1][0] * x[0][0])
    J[0, 2] = 3.0 * np.exp(- (3.0 * x[2][0]))
    J[0, 3] = x[4][0] * x[4][0]
    J[0, 4] = 2.0 * x[3][0] * x[4][0]
    J[0, 5] = -1.0
    J[0, 6] = 0.0
    J[0, 7] = -2.0 * np.cosh(2.0 * x[7][0]) * x[8][0]
    J[0, 8] = -np.sinh(2.0 * x[7][0])
    J[0, 9] = 2.0

    J[1, 0] = x[1][0] * np.cos(x[1][0] * x[0][0])
    J[1, 1] = x[0][0] * np.cos(x[1][0] * x[0][0])
    J[1, 2] = x[8][0] * x[6][0]
    J[1, 3] = 0.0
    J[1, 4] = 6.0 * x[4][0]
    J[1, 5] = -np.exp(-x[9][0] + x[5][0]) - x[7][0] - 1.0
    J[1, 6] = x[2][0] * x[8][0]
    J[1, 7] = -x[5][0]
    J[1, 8] = x[2][0] * x[6][0]
    J[1, 9] = np.exp(-x[9][0] + x[5][0])

    J[2] = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1])

    J[3, 0] = - x[4][0] * np.power(x[2][0] + x[0][0], -2.0)
    J[3, 1] = -2.0 * x[1][0] * np.cos(x[1][0] * x[1][0])
    J[3, 2] = - x[4][0] * np.power(x[2][0] + x[0][0], -2.0)
    J[3, 3] = -2.0 * np.sin(-x[8][0] + x[3][0])
    J[3, 4] = 1.0 / (x[2][0] + x[0][0])
    J[3, 5] = 0
    J[3, 6] = -2.0 * np.cos(x[6][0] * x[9][0]) * x[9][0] * np.sin(x[6][0] * x[9][0])
    J[3, 7] = -1
    J[3, 8] = 2.0 * np.sin(-x[8, 0] + x[3, 0])
    J[3, 9] = -2.0 * np.cos(x[6, 0] * x[9, 0]) * x[6, 0] * np.sin(x[6, 0] * x[9, 0])
    J[4, 0] = 2 * x[7, 0]
    J[4, 1] = -2.0 * np.sin(x[1, 0])
    J[4, 2] = 2 * x[7, 0]
    J[4, 3] = np.power(-x[8, 0] + x[3, 0], -2.0)
    J[4, 4] = np.cos(x[4, 0])
    J[4, 5] = x[6, 0] * np.exp(-x[6, 0] * (-x[9, 0] + x[5, 0]))
    J[4, 6] = -(x[9, 0] - x[5, 0]) * np.exp(-x[6, 0] * (-x[9, 0] + x[5, 0]))
    J[4, 7] = 2 * x[2, 0] + 2.0 * x[0, 0]
    J[4, 8] = -np.power(-x[8, 0] + x[3, 0], -2.0)
    J[4, 9] = -x[6, 0] * np.exp(-x[6, 0] * (-x[9, 0] + x[5, 0]))
    J[5, 0] = np.exp(x[0, 0] - x[3, 0] - x[8, 0])
    J[5, 1] = -3.0 / 2.0 * x[9, 0] * np.sin(3.0 * x[9, 0] * x[1, 0])
    J[5, 2] = -x[5, 0]
    J[5, 3] = -np.exp(x[0, 0] - x[3, 0] - x[8, 0])
    J[5, 4] = 2 * x[4, 0] / x[7, 0]
    J[5, 5] = -x[2, 0]
    J[5, 6] = 0
    J[5, 7] = -x[4, 0] * x[4, 0] * np.power(x[7, 0], -2)
    J[5, 8] = -np.exp(x[0, 0] - x[3, 0] - x[8, 0])
    J[5, 9] = -3.0 / 2.0 * x[1, 0] * np.sin(3.0 * x[9, 0] * x[1, 0])
    J[6, 0] = np.cos(x[3, 0])
    J[6, 1] = 3.0 * x[1, 0] ** 2 * x[6, 0]
    J[6, 2] = 1
    J[6, 3] = -(x[0, 0] - x[5, 0]) * np.sin(x[3, 0])
    J[6, 4] = x[9, 0] * (x[4, 0] ** (-2)) * np.cos(x[9, 0] / x[4, 0] + x[7, 0])
    J[6, 5] = -np.cos(x[3, 0])
    J[6, 6] = x[1, 0] ** 3
    J[6, 7] = -np.cos(x[9, 0] / x[4, 0] + x[7, 0])
    J[6, 8] = 0
    J[6, 9] = -1.0 / x[4, 0] * np.cos(x[9, 0] / x[4, 0] + x[7, 0])
    J[7, 0] = 2.0 * x[4, 0] * (x[0, 0] - 2.0 * x[5, 0])
    J[7, 1] = -x[6, 0] * np.exp(x[1, 0] * x[6, 0] + x[9, 0])
    J[7, 2] = -2.0 * np.cos(-x[8, 0] + x[2, 0])
    J[7, 3] = 1.5
    J[7, 4] = (x[0, 0] - 2.0 * x[5, 0]) ** 2
    J[7, 5] = -4.0 * x[4, 0] * (x[0, 0] - 2.0 * x[5, 0])
    J[7, 6] = -x[1, 0] * np.exp(x[1, 0] * x[6, 0] + x[9, 0])
    J[7, 7] = 0
    J[7, 8] = 2.0 * np.cos(-x[8, 0] + x[2, 0])
    J[7, 9] = -np.exp(x[1, 0] * x[6, 0] + x[9, 0])
    J[8, 0] = -3
    J[8, 1] = -2.0 * x[7, 0] * x[9, 0] * x[6, 0]
    J[8, 2] = 0
    J[8, 3] = np.exp(x[4, 0] + x[3, 0])
    J[8, 4] = np.exp(x[4, 0] + x[3, 0])
    J[8, 5] = -7.0 * x[5, 0] ** (-2)
    J[8, 6] = -2.0 * x[1, 0] * x[7, 0] * x[9, 0]
    J[8, 7] = -2.0 * x[1, 0] * x[9, 0] * x[6, 0]
    J[8, 8] = 3
    J[8, 9] = -2.0 * x[1, 0] * x[7, 0] * x[6, 0]
    J[9, 0] = x[9, 0]
    J[9, 1] = x[8, 0]
    J[9, 2] = -x[7, 0]
    J[9, 3:6] = np.cos(x[3:6, 0].sum()) * x[6, 0]
    J[9, 6] = np.sin(x[3:6, 0].sum())
    J[9, 7] = -x[2, 0]
    J[9, 8] = x[1, 0]
    J[9, 9] = x[0, 0]
    return J

def norm(Matrix):
    row, col = Matrix.shape
    max_norm = 0
    tmp_norm = 0
    for i in range(col):
        max_norm += abs(Matrix[0, i])

    for i in range(1, row):
        for j in range(col):
            tmp_norm += abs(Matrix[i, j])
        if tmp_norm > max_norm:
            max_norm = tmp_norm
        tmp_norm = 0

    return max_norm


def LU_dec(L, U, P, counter):
    n = U.shape[0]

    for j in range(0, n):
        max_row = np.argmax(np.abs(U[j:, j])) + j
        counter += 5  # Ищем минимальный элемент. Примерно 5 сравнений.
        L[[j, max_row]] = L[[max_row, j]]
        P[[j, max_row]] = P[[max_row, j]]
        U[[j, max_row]] = U[[max_row, j]]
        counter += 30  # Меняем строки местами. 10 присваиваний(3 раза).

        for i in range(j, n):
            L[i][j] = U[i][j] / U[j][j]
            counter += 1

        for k in range(j + 1, n):
            for r in range(j, n):
                U[k][r] = U[k][r] - U[j][r] * L[k][j]
                counter += 2

    return L, U, P, counter


def SLAE(L, U, P, b, counter):
    n = U.shape[0]
    Pb = np.dot(P, b)
    counter += 10
    y = np.zeros((n, 1))
    x = np.zeros((n, 1))

    for i in range(n):
        y[i][0] = Pb[i][0]
        for j in range(i):
            y[i][0] -= y[j][0] * L[i][j]
        y[i][0] /= L[i][i]
        counter += 4

    for i in range(n - 1, -1, -1):
        x[i][0] = y[i][0]
        for j in range(n - 1, i, -1):
            x[i][0] -= x[j][0] * U[i][j]
        x[i][0] /= U[i][i]
        counter += 4

    return x


def newton(x_0, err):
    start = time.time()
    count = 0
    x = np.array(x_0)
    L = np.zeros((10, 10))
    P = np.identity(10)

    for i in range(1000):
        L.fill(0)
        P.fill(0)
        np.fill_diagonal(P, 1)
        U = jacobi(x)
        L, U, P, count = LU_dec(L, U, P, count)
        x_del = SLAE(L, U, P, -1 * function(x), count)
        x = x_del + x
        if norm(x_del) < err:
            end = time.time()
            print("1) {} iterations".format(i))
            print("2) {} milliseconds".format(end - start))
            print("3) {} operation for LU decomp".format(count))
            return x

    end = time.time()
    print("1) {} iterations".format(1000))
    print("2) {} milliseconds".format(end - start))
    print("3) {} operation for LU decomp".format(count))
    return x


def mod_newton(x_0, err):
    start = time.time()
    count = 0
    x = np.array(x_0)
    L = np.zeros((10, 10))
    U = jacobi(x_0)
    P = np.eye(10)
    L, U, P, count = LU_dec(L, U, P, count)

    for i in range(1000):
        f = function(x)
        x_del = SLAE(L, U, P, -1 * f, count)
        x = x_del + x
        if norm(x_del) < err and norm(f) < err:
            end = time.time()
            print("1) {} iterations".format(i))
            print("2) {} milliseconds".format(end - start))
            print("3) {} operations for LU decomp".format(count))
            return x
    end = time.time()
    print("1) {} iterations".format(1000))
    print("2) {} milliseconds".format(end - start))
    print("3) {} operation for LU decomp".format(count))
    return x


def hyb_newton(x_0, k, err):
    start = time.time()
    count = 0

    x = np.array(x_0)
    L = np.zeros((10, 10))
    U = jacobi(x_0)
    P = np.eye(10)

    for i in range(1000):
        if i < k:
            P = np.eye(10)
            L = np.zeros((10, 10))
            U = jacobi(x)
            L, U, P, count = LU_dec(L, U, P, count)

        f = function(x)
        x_del = SLAE(L, U, P, -1 * f, count)
        x = x_del + x
        if norm(x_del) < err and norm(f) < err:
            end = time.time()
            print("1) {} iterations".format(i))
            print("2) {} milliseconds".format(end - start))
            print("3) {} operations for LU decomp".format(count))
            print(function(x))
            return x
    end = time.time()
    print("1) {} iterations".format(1000))
    print("2) {} milliseconds".format(end - start))
    print("3) {} operations for LU decomp".format(count))
    print(function(x))
    return x


#def hyb_newton(x_0, err):
#    start = time.time()
#    count = 0

#    x = np.copy(x_0)
#    L = np.zeros((10,10))
#    U = jacobi(x_0)
#    P = np.eye(10)
#    L, U, P, count = LU_dec(L, U, P, count)

#    for i in range(1000):
#        if i % 3 == 0:
#            P = np.eye(10)
#            L = np.zeros((10, 10))
#            U = jacobi(x)
#            L, U, P, count = LU_dec(L, U, P, count)

#        f = function(x)
#        x_del = SLAE(L, U, P, -1 * f)
#        x = x_del + x
#        if norm(x_del) < err and norm(f) < err:
#            end = time.time()
#            print("1) {} iterations".format(i))
#            print("2) {} milliseconds".format(end - start))
#            print("3) {} operations for LU decomp".format(count))
#            return x
#    end = time.time()
#    print("1) {} iterations".format(1000))
#    print("2) {} milliseconds".format(end - start))
#    print("3) {} operations for LU decomp".format(count))
#    return x


err = 1e-13
# x = np.array([0.5, 0.5, 1.5, -1.0, -0.2, 1.5, 0.5, -0.5, 1.5, -1.5])
x = np.zeros((10,1))
x[0][0] = 0.5;
x[1][0] = 0.5;
x[2][0] = 1.5;
x[3][0] = -1.0;
x[4][0] = -0.2;
x[5][0] = 1.5;
x[6][0] = 0.5;
x[7][0] = -0.5;
x[8][0] = 1.5;
x[9][0] = -1.5;

print('\n====\tNewton Method\t====\n')
x_newton = newton(x, err)
print(function(x_newton))
print('x = \n', x_newton)

print('\n====\tMod Newton Method\t====\n')
x_mod_newton = mod_newton(x, err)
print('x = \n', x_mod_newton)

k = 3
print('\n====\tHybrid Method Before {} iterations\t====\n'.format(k))
x_hyb_newton = hyb_newton(x, k, err)
print('x = \n', x_hyb_newton)

k = 4
print('\n====\tHybrid Method Before {} iterations\t====\n'.format(k))
x_hyb_newton = hyb_newton(x, k, err)
print('x = \n', x_hyb_newton)

k = 5
print('\n====\tHybrid Method Before {} iterations\t====\n'.format(k))
x_hyb_newton = hyb_newton(x, k, err)
print('x = \n', x_hyb_newton)

k = 6
print('\n====\tHybrid Method Before {} iterations\t====\n'.format(k))
x_hyb_newton = hyb_newton(x, k, err)
print('x = \n', x_hyb_newton)

k = 7
print('\n====\tHybrid Method Before {} iterations\t====\n'.format(k))
x_hyb_newton = hyb_newton(x, k, err)
print('x = \n', x_hyb_newton)

k = 8
print('\n====\tHybrid Method Before {} iterations\t====\n'.format(k))
x_hyb_newton = hyb_newton(x, k, err)
print('x = \n', x_hyb_newton)

# print('\n====\tHybrid Method every 10 iterations\t====\n')
# x_hyb_newton = hyb_newton(x,k, err)
# print('x = \n', x_hyb_newton)