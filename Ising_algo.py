import numpy as np
import matplotlib.pyplot as plt
import time
import copy as cp
from utils import quad_K,verif_3sat

# parameters for sat_k3_v64_c192.cnf
ALPHA = 1.0
BETA = 1.0
GAMMA = BETA / 2
std = 6.5

# threshold is 3/2 of the K matrix row sum and subtract one half of the matrix Q diagonal elements where K matrix is ALPHA*I + BETA*Q,
def getThresholds(K, L):
    matrix = np.zeros((K.shape[0]))
    for i in range(K.shape[0]):
        matrix[i] = (ALPHA + K[i].sum() * BETA)/2 - L[i]*GAMMA
    return matrix

def SvectorInitialization(S):
    for i in range(S.shape[0]):
        val = np.random.randint(0, 2 ** 15) % 2
        S[i] = val

    return S

# (1-S) x S_PIC, which is actually partial sum of S_PIC
# S and S_PIC are both 1-dim vector
def calculateEnergy(S, S_PIC, L, C):
    energy = 0
    for i in range(S.shape[0]):
        if S[i] == 0:
            energy += S_PIC[i]
    for i in range(S.shape[0]):
        if S[i] == 1:
            energy -= L[i]

    energy = 2*energy+C

    return energy

def calculateEnergy_II(S, S_PIC, L, P, C):
    energy = 0
    for i in range(S.shape[0]):
        if S[i] == 1:
            energy += (P[i]-S_PIC[i]-L[i])
    energy = 2*energy+C

    return energy

def isVectorZero(S):
    return np.all(S == 0)


def isVectorOne(S):
    return np.all(S == 1)

def compareToThresholds(S, thresholds):
    n = S.shape[0]
    for i in range(n):
        if S[i] > thresholds[i]:
            S[i] = 1
        else:
            S[i] = 0


def runIsingAlgorithm_simulator(K, L, niter, MATRIX_SIZE, cnfs, n):
    # This function Run the Ising algorithm
    # Input argument
    # K: the K matrix of the QUBO problems
    # L: the L matrix of the QUBO problems
    # C: the constant of the QUBO
    # niter: the maximum iterations
    # output argument
    # best_candidate: the best state of the spins

    class BestCandidate:
        matrix = np.zeros((MATRIX_SIZE, 1), dtype=np.int)
        energy = 0
    best_candidate = BestCandidate
    # Initialization stats S and energy
    S = np.zeros((MATRIX_SIZE), dtype=np.int)
    SvectorInitialization(S)
    eng = np.zeros(niter)
    # Set Matrix
    M = ALPHA*np.eye(MATRIX_SIZE)+BETA*K
    scale, MS = quad_K(M)
    print('scale',scale)
    print('MS',MS)
    # Calculate initial energy
    energy = verif_3sat(cnfs, S.T[0:n])
    best_candidate.matrix = S
    best_candidate.energy = energy
    # Using the adjacency matrix to set the thresholds
    thresholds = getThresholds(K,L)
    startTime = time.perf_counter()
    # start the iteration
    for i in range(niter):
        ##Calculate matrix multiplication and energy
        S_PIC = MS @ S
        for j in range(K.shape[0]):
            noise = np.random.randn() * std
            S_PIC[j] = S_PIC[j] + noise
        S_PIC = S_PIC/scale
        energy = verif_3sat(cnfs, S.T[0:n])
        if energy > best_candidate.energy and not isVectorZero(S) and not isVectorOne(S):
            best_candidate.matrix = cp.deepcopy(S)
            best_candidate.energy = cp.deepcopy(energy)

            print("New best")
            print(f"Iteration {i}")
            print(best_candidate.matrix.T)
            print(best_candidate.energy)

        # Updata the state S
        S = S_PIC
        # compare the state S with the thresholds
        compareToThresholds(S, thresholds)
        S = S.astype(int)
    endTime = time.perf_counter()
    d = endTime - startTime
    print("finished iterations ")
    print("time only for loops:")
    print(f"caling {i} iterations: {d} s")
    print(f"average time per 5000 iterations: {d * 5000.0 / i} s")

    return best_candidate