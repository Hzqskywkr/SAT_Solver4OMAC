import numpy as np
import matplotlib.pyplot as plt
import time
import copy as cp
from utils import quad_K,verif_3sat

VERBOSE = 0
s = 2 ** (0)
g = 6.5

# threshold is 3/2 of the K matrix row sum and subtract one half of the matrix Q diagonal elements where K matrix is ALPHA*I + BETA*Q,
def getThresholds(K, L):
    matrix = np.zeros((K.shape[0]))
    for i in range(K.shape[0]):
        matrix[i] = K[i].sum() - L[i]
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
            energy += P[i]-S_PIC[i]-L[i]
    energy = 2*energy+C

    return energy

def isVectorZero(S):
    return np.all(S == 0)


def isVectorOne(S):
    return np.all(S == 1)

def findminindex(DH):
    minDH = np.min(DH)
    index_minSigma = []
    for i in range(DH.shape[0]):
        if DH[i] == minDH:
            index_minSigma.append(i)
    return index_minSigma

def Flip(S,S_PIC, thresholds):
    Sigma = 2 * S - 1
    DM = 2 * S_PIC.T - thresholds
    DH = 2 * Sigma.T * DM  # (-2)*Sigma'*DM
    minSigma = findminindex(DH)
    index = np.random.choice(minSigma)
    Sigma[index] = -Sigma[index]
    S = (Sigma + 1) / 2
    return S

def runAGHNNIsingAlgorithm_quad(K, L, niter, MATRIX_SIZE, cnfs, n):

    class BestCandidate:
        matrix = np.zeros((MATRIX_SIZE, 1), dtype=np.int)
        energy = 0
    best_candidate = BestCandidate
    # Initialization stats S and energy
    S = np.zeros((MATRIX_SIZE), dtype=np.int)
    SvectorInitialization(S)
    eng = np.zeros((niter, 1), dtype=np.int)
    #print("initial S:")
    #print(S.T)
    if VERBOSE:
        print("initial S:")
        print(S.T)
    # Calculate initial energy
    scale, KS = quad_K(K)
    print('scale', scale)
    print('KS', KS)
    energy = verif_3sat(cnfs, S.T[0:n])
    best_candidate.matrix = S
    best_candidate.energy = energy

    #print(f"initial energy = {energy}")
    # Using the adjacency matrix to set the thresholds
    thresholds = getThresholds(K, L)
    if VERBOSE:
        print("thresholds :")
        print(thresholds.T)

    for i in range(niter):
        if VERBOSE:
            print(f"--------Iteration-----{i}--")
            print(f"S Matrix : {S.T}")
            print(f"K Matrix : {K}")
            print(f"L Vector : {L}")

        ##Calculate matrix multiplication and energy
        S_PIC = KS @ S
        # add noise to S_PIC
        for j in range(K.shape[0]):
            noise = np.random.randn() * g
            S_PIC[j] = S_PIC[j] + noise
        S_PIC = S_PIC/scale
        if VERBOSE:
            print("S_PIC :")
            print(S_PIC.T)


        energy = verif_3sat(cnfs, S.T[0:n])
        eng[i] = energy
        # update the best energy
        if energy < best_candidate.energy and not isVectorZero(S) and not isVectorOne(S):
            best_candidate.matrix = cp.deepcopy(S)
            best_candidate.energy = energy
            print("New best")
            print(f"Iteration {i}")
            print(best_candidate.matrix.T)
            print(best_candidate.energy)

        # Updata the state S
        S = Flip(S, S_PIC, thresholds)
        S = S.astype(int)
        if VERBOSE:
            print("S' :")
            print(S.T)

    return best_candidate, eng