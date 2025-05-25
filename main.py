import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math

def readFromFile(filename):
    data = pd.read_csv(filename, header=None)
    return data[0].to_numpy(), data[1].to_numpy() #x, y

first_x, first_y = readFromFile("2018_paths/100.csv")
second_x, second_y = readFromFile("2018_paths/SpacerniakGdansk.csv")
third_x, third_y = readFromFile("2018_paths/chelm.csv")
fourth_x, fourth_y = readFromFile("2018_paths/GlebiaChallengera.csv")

figure, axis = plt.subplots(2)
axis[0].plot(first_x, first_y)
axis[0].set_title("100")
axis[0].set_xlabel("x [m]")
axis[0].set_ylabel("y [m]")

axis[1].plot(second_x, second_y)
axis[1].set_title("Spacerniak Gdańsk")
axis[1].set_xlabel("x [m]")
axis[1].set_ylabel("y [m]")
"""
axis[0, 1].plot(third_x, third_y)
axis[0, 1].set_title("Wielki Kanion Kolorado")
axis[0, 1].set_xlabel("x [m]")
axis[0, 1].set_ylabel("y [m]")

axis[1, 1].plot(fourth_x, fourth_y)
axis[1, 1].set_title("Redlujjj")
axis[1, 1].set_xlabel("x [m]")
axis[1, 1].set_ylabel("y [m]")
"""
plt.subplots_adjust(hspace=0.6)
plt.show()

#interpolation
#interpolate the given data and create plots

def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)    
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr

def LangrangeInterpolation(x, y, xi, N):
    #take exactly n evenly spaced out elements from array
    idx = np.round(np.linspace(0, len(x) - 1, N)).astype(int)
    inter_x = x[idx]
    inter_y = y[idx]
    inter_x = normalize(inter_x, 0, 1)
    n = len(inter_y)
    result = 0.0
    for i in range(n):

        # Compute individual terms of above formula
        term = inter_y[i]
        for j in range(n):
            if j != i:
                term = term * (xi - inter_x[j]) / (inter_x[i] - inter_x[j])

        # Add current term to result
        result += term

    return result

def plotLagrange(x,y,N=128,name="default"):
    plot_min_number = np.round(np.log2(N)-3).astype(int)
    #figure, axis = plt.subplots(4)
    #figure.suptitle(f"Interpolacja metodą Lagrange'a trasy {name} stosując N węzłów")
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
#plotLagrange(third_x, third_y, N=128, name="Wielki Kanion")

def splineInterpolation(x, y, xi, N):
    idx = np.round(np.linspace(0, len(x) - 1, N)).astype(int)
    inter_x = x[idx]
    inter_y = y[idx]
    inter_x = normalize(inter_x, 0, 1)
    h = 1/N
    A = np.array([[1,0,0,0,0,0,0,0],
                  [0,0,0,0,1,0,0,0],
                  [1,h,h*h,h*h*h,0,0,0,0],
                  [0,0,0,0,1,h,h*h,h*h*h],
                  [0,1,2*h,3*h*h,0,-1,0,0],
                  [0,0,2,6*h,0,0,-2,0],
                  [0,0,h,0,0,0,0,0],
                  [0,0,0,0,0,0,2,6*h]])
    i = 0
    while(xi > inter_x[i+1]):
        i += 1
    if xi > inter_x[-2]:
        i -= 1
        b = np.array([[inter_y[i]],
                    [inter_y[i+1]],
                    [inter_y[i+1]],
                    [inter_y[i+2]],
                    [0],
                    [0],
                    [0],
                    [0]])
        c = np.linalg.solve(A,b)
        dx = xi - inter_x[i+1]
        result = c[4][0] + c[5][0]*dx + c[6][0]*dx**2 + c[7][0]*dx**3
        return float(result)
    else:
        b = np.array([[inter_y[i]],
                    [inter_y[i+1]],
                    [inter_y[i+1]],
                    [inter_y[i+2]],
                    [0],
                    [0],
                    [0],
                    [0]])
        c = np.linalg.solve(A,b)
        dx = xi - inter_x[i]
        result = c[0][0] + c[1][0]*dx + c[2][0]*dx**2 + c[3][0]*dx**3
        return float(result)

def plotSpline(x,y,N=128,name="default"):
    plot_min_number = np.round(np.log2(N)-3).astype(int)
    #figure, axis = plt.subplots(4)
    #figure.suptitle(f"Interpolacja metodą Splajnów trzeciego stopnia trasy {name} stosując N węzłów")
    for row in range(4):
        n = 2**(plot_min_number + row)
        domain_x = np.linspace(0.0, 1.0, len(x))
        domain_y = [splineInterpolation(x,y,domain_x[i], n) for i in range(len(x))]
        domain_x = normalize(domain_x, 0.0, x[-1])
        plt.plot(domain_x, domain_y)
        plt.plot(x,y)
        idx = np.round(np.linspace(0, len(x) - 1, n)).astype(int)
        plt.scatter(x[idx],y[idx])
        plt.ylim(np.min(y) * 0.75, np.max(y) * 1.5)
        plt.legend(["interpolowane","oryginalne","węzły"])
        plt.title(f"Interpolacja metodą Splajnów trzeciego stopnia trasy {name} stosując N = {n} węzłów")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.subplots_adjust(hspace=0.6)
        plt.show()

#plotSpline(first_x, first_y, N=128, name="100")
#plotSpline(second_x, second_y, N=128, name="Spacerniak Gdańsk")
"""
def getChebyshev0to1(n):
    nodes = np.array([ math.cos(((k-0.5)*math.pi)/n) for k in range(1,n+1) ])
    nodes += 1
    nodes = np.flip(normalize(nodes, 0, 1))
    return nodes

def LangrangeInterpolationChebyshev(x, y, xi, N):
    #take exactly n evenly spaced out elements from array
    idx = np.round(getChebyshev0to1(N)*(len(x)-1)).astype(int)
    inter_x = x[idx]
    inter_y = y[idx]
    inter_x = normalize(inter_x, 0, 1)
    n = len(inter_y)
    result = 0.0
    for i in range(n):

        # Compute individual terms of above formula
        term = inter_y[i]
        for j in range(n):
            if j != i:
                term = term * (xi - inter_x[j]) / (inter_x[i] - inter_x[j])

        # Add current term to result
        result += term

    return result

def plotLagrangeChebyshev(x,y,N=128,name="default"):
    plot_min_number = np.round(np.log2(N)-3).astype(int)
    figure, axis = plt.subplots(4)
    figure.suptitle(f"Interpolacja metodą Lagrange'a trasy {name} stosując N węzłów")
    for row in range(4):
        n = 2**(plot_min_number + row)
        domain_x = np.linspace(0.0, 1.0, len(x))
        domain_y = [LangrangeInterpolation(x,y,domain_x[i], n) for i in range(len(x))]
        domain_x = normalize(domain_x, 0.0, x[-1])
        axis[row].plot(domain_x, domain_y)
        axis[row].plot(x,y)
        idx = np.round(getChebyshev0to1(n)*(len(x)-1)).astype(int)
        axis[row].scatter(x[idx],y[idx])
        axis[row].set_ylim(np.min(y) * 0.75, np.max(y) * 1.5)
        axis[row].legend(["interpolowane","oryginalne","węzły"])
        axis[row].set_title(f"N = {n}")
        axis[row].set_xlabel("x [m]")
        axis[row].set_ylabel("y [m]")
    plt.subplots_adjust(hspace=0.6)
    plt.show()

plotLagrangeChebyshev(first_x, first_y, N=64, name="100")
#plotLagrangeChebyshev(second_x, second_y, N=64, name="100")
"""
