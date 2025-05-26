import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def readFromFile(filename):
    data = pd.read_csv(filename, header=None)
    return data[0].to_numpy(), data[1].to_numpy() #x, y

first_x, first_y = readFromFile("2018_paths/100.csv")
second_x, second_y = readFromFile("2018_paths/SpacerniakGdansk.csv")

figure, axis = plt.subplots(2)
axis[0].plot(first_x, first_y)
axis[0].set_title("100")
axis[0].set_xlabel("x [m]")
axis[0].set_ylabel("y [m]")

axis[1].plot(second_x, second_y)
axis[1].set_title("Spacerniak Gdańsk")
axis[1].set_xlabel("x [m]")
axis[1].set_ylabel("y [m]")

plt.subplots_adjust(hspace=0.6)
plt.show()

#interpolation

def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)    
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr

def LangrangeInterpolation(x, y, xi, N):
    idx = np.round(np.linspace(0, len(x) - 1, N)).astype(int)
    inter_x = x[idx]
    inter_y = y[idx]
    inter_x = normalize(inter_x, 0, 1)
    length = len(inter_y)
    result = 0.0
    for i in range(length):
        term = inter_y[i]
        for j in range(length):
            if j != i:
                term = term * (xi - inter_x[j]) / (inter_x[i] - inter_x[j])
        result += term
    return result

def plotLagrange(x,y,N=128,name="default"):
    plot_min_number = np.round(np.log2(N)-3).astype(int)
    for row in range(4):
        n = 2**(plot_min_number + row)
        domain_x = np.linspace(0.0, 1.0, len(x))
        domain_y = [LangrangeInterpolation(x,y,domain_x[i], n) for i in range(len(x))]
        domain_x = normalize(domain_x, 0.0, x[-1])
        plt.plot(domain_x, domain_y)
        plt.plot(x,y)
        idx = np.round(np.linspace(0, len(x) - 1, n)).astype(int)
        plt.scatter(x[idx],y[idx])
        plt.ylim(np.min(y) * 0.75, np.max(y) * 1.5)
        plt.legend(["interpolowane","oryginalne","węzły"])
        plt.title(f"Interpolacja metodą Lagrange'a trasy {name} stosując N = {n} węzłów")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.subplots_adjust(hspace=0.6)
        plt.show()

#plotLagrange(first_x, first_y, N=128, name="100")
#plotLagrange(second_x, second_y, N=128, name="Spacerniak Gdańsk") #<----------------------------------------

def splineInterpolation(x, y, xi, N):
    idx = np.round(np.linspace(0, len(x) - 1, N)).astype(int)
    idx = np.unique(idx)
    inter_x = x[idx]
    inter_y = y[idx]
    n = len(inter_x)

    #if last interval then handle differently
    if xi == x[-1]:
        return y[-1]
    if xi >= inter_x[-2]:
        i = n - 3
        use_second_poly = True
    else:
        i = 0
        while i < n - 2 and xi > inter_x[i + 1]:
            i += 1
        use_second_poly = xi > inter_x[i + 1]
    #calc distance between nodes
    h = inter_x[i + 1] - inter_x[i]
    A = np.array([
        [1, 0,     0,     0,     0, 0,     0,     0],
        [0, 0,     0,     0,     1, 0,     0,     0],
        [1, h,   h**2,  h**3,    0, 0,     0,     0],
        [0, 0,     0,     0,     1, h,   h**2,  h**3],
        [0, 1,  2*h,  3*h**2,    0, -1,     0,     0],
        [0, 0,     2,  6*h,      0,  0,    -2,     0],
        [0, 0,    h,     0,      0,  0,     0,     0],
        [0, 0,     0,     0,     0,  0,     2,  6*h]
    ])
    b = np.array([
        [inter_y[i]],
        [inter_y[i + 1]],
        [inter_y[i + 1]],
        [inter_y[i + 2]],
        [0],
        [0],
        [0],
        [0]
    ])
    c = np.linalg.solve(A, b)
    if use_second_poly:
        dx = xi - inter_x[i + 1]
        result = c[4][0] + c[5][0]*dx + c[6][0]*dx**2 + c[7][0]*dx**3
    else:
        dx = xi - inter_x[i]
        result = c[0][0] + c[1][0]*dx + c[2][0]*dx**2 + c[3][0]*dx**3
    return float(result)


def plotSpline(x,y,N=128,name="default"):
    plot_min_number = np.round(np.log2(N)-3).astype(int)
    for row in range(4):
        n = 2**(plot_min_number + row)
        domain_x = np.linspace(0.0, x[-1], len(x))
        domain_y = [splineInterpolation(x,y,domain_x[i], n) for i in range(len(x))]
        plt.plot(domain_x, domain_y)
        plt.plot(x,y)
        idx = np.round(np.linspace(0, len(x) - 1, n)).astype(int)
        idx = np.unique(idx)
        plt.scatter(x[idx],y[idx])
        plt.ylim(np.min(y) * 0.75, np.max(y) * 1.5)
        plt.legend(["interpolowane","oryginalne","węzły"])
        plt.title(f"Interpolacja metodą Splajnów trzeciego stopnia trasy {name} stosując N = {n} węzłów")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.subplots_adjust(hspace=0.6)
        plt.show()

plotSpline(first_x, first_y, N=128, name="100") #<-----------------------------------------------
plotSpline(second_x, second_y, N=128, name="Spacerniak Gdańsk")

def Lagrangeinterpolation2(x, y, xi):
    length = len(x)
    result = 0
    for i in range(length):
        term = y[i]
        for j in range(length):
            if j != i:
                term *= (xi - x[j]) / (x[i] - x[j])
        result += term
    return result

def generateChebyshevNodes(n, length):
    nodes = []
    for k in range(n):
        x_k = np.cos((2 * k + 1) * np.pi / (2 * n))
        index = int(round(0.5 * (x_k + 1) * (length - 1)))
        nodes.append(index)
    return np.array(nodes)

def plotLagrangeChebyshev(x, y, name, N=16):
    plot_min_number = np.round(np.log2(N)).astype(int)
    for row in range(4):
        n = 2**(plot_min_number + row)
        chebyshev_indices = generateChebyshevNodes(n, len(x))
        chebyshev_indices = np.unique(chebyshev_indices)

        nodes_x = x[chebyshev_indices]
        nodes_x /= np.max(nodes_x)
        nodes_y = y[chebyshev_indices]

        inter_x = np.linspace(0.0, 1.0, len(x))
        inter_y = [Lagrangeinterpolation2(nodes_x, nodes_y, x) for x in inter_x]

        inter_x *= np.max(x)
        nodes_x *= np.max(x)

        plt.plot(inter_x, inter_y)
        plt.plot(x, y)
        plt.scatter(nodes_x, nodes_y)
        plt.ylim(np.min(y) * 0.75, np.max(y) * 1.5)
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.title(f"Interpolacja metodą Lagrange'a trasy {name} stosując N = {n} węzłów Chebysheva")
        plt.legend(["interpolowane","oryginalne","węzły"])
        plt.show()

#plotLagrangeChebyshev(second_x,second_y,"Gdansk") #<--------------------------------------------
