import numpy as np
import time
import copy

std = 6.5

def randVector(n, s):
    for i in range(n):
        val = np.random.randint(0, 2 ** 15) % 2
        s[i] = val

def SvectorInitialization(S):
    for i in range(S.shape[0]):
        val = np.random.randint(0, 2 ** 15) % 2
        S[i] = 2*val - 1

    return S

def Detect_unsat_clauses(Y):
    clause_M = np.array(np.where(Y[:] <= 0)).ravel()
    num_unsatclauses = len(clause_M)
    return clause_M, num_unsatclauses

def findmaxindex(DH):
    maxDH = np.max(DH)
    index_maxSigma = []
    for i in range(3):
        if DH[i] == maxDH:
            index_maxSigma.append(i)
    return index_maxSigma

def update_simp(abs_cnfs_1, Matrix_cnf, Thresholds, unsatclauses, Spin):
    ran_clause = np.random.choice(unsatclauses)
    ch_absclause_1 = abs_cnfs_1[ran_clause]
    DH_clause = []
    s3 = 3
    for var in ch_absclause_1:
        energy_temp = 0
        Spin_change = copy.deepcopy(Spin)
        Spin_change[var] = - Spin_change[var]
        vect = Matrix_cnf @ (Spin_change*s3)
        for i in range(vect.shape[0]):
            noise = np.random.randn() * std
            vect[i] += noise
            vect[i] = vect[i]/s3 - Thresholds[i]
            if vect[i] > 0:
                energy_temp += 1
        DH_clause.append(energy_temp)
    index_maxSigma = findmaxindex(DH_clause)
    index = np.random.choice(index_maxSigma)
    Flip_index = ch_absclause_1[index]
    Spin[Flip_index] = - Spin[Flip_index]

    return Spin

def runGSRWAlgorithm(abs_cnfs_1, m, Matrix_cnf, Thresholds, MATRIX_SIZE, niter):

    class BestCandidate:
        matrix = np.zeros((MATRIX_SIZE, 1), dtype=np.int)
        energy = 0
    best_candidate = BestCandidate
    Spin = np.zeros(MATRIX_SIZE, dtype=np.int)
    SvectorInitialization(Spin)
    # Calculate initial energy
    S_PIC = Matrix_cnf @ Spin
    Y = S_PIC - Thresholds
    unsatclauses, num_unsatclauses = Detect_unsat_clauses(Y)
    best_candidate.matrix = Spin
    best_candidate.energy = len(unsatclauses)
    startTime = time.perf_counter()
    s2 = 6
    # start the iteration
    for i in range(niter):
        S_PIC = Matrix_cnf @ (Spin*s2)
        for l in range(m):
            noise = np.random.randn() * std
            S_PIC[l] += noise
        S_PIC = S_PIC / s2
        Y = S_PIC - Thresholds
        # Calculate energy and update the best energy
        unsatclauses, num_unsatclauses = Detect_unsat_clauses(Y)
        if num_unsatclauses < best_candidate.energy:
            best_candidate.matrix = Spin.copy()
            best_candidate.energy = num_unsatclauses
            print("New best")
            print(f"Iteration {i}")
            print(best_candidate.matrix.T)
            print(best_candidate.energy)

        if best_candidate.energy == 0:
            break
        # Updata the state S
        Spin = update_simp(abs_cnfs_1, Matrix_cnf, Thresholds, unsatclauses, Spin)
        Spin = Spin.astype(int)
    endTime = time.perf_counter()
    d = endTime - startTime
    print("finished iterations ")
    print("time only for loops:")
    print(f"caling {i} iterations: {d} s")
    print(f"average time per 5000 iterations: {d * 5000.0 / i} s")

    return best_candidate