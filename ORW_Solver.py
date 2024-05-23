import numpy as np
import time
from utils import read_cnf_GSRW, verif_3sat_v2
from ORW_algo import runGSRWAlgorithm

def main():

    seed = int(time.time())
    np.random.seed(seed)
    NITER = 5000
    cnf_files = ['sat_k3_v16_c48_s431343124.cnf', 'sat_k3_v64_c192.cnf']
    cnf_inst = 1
    cnfs, abs_cnfs_1, n, m = read_cnf_GSRW(cnf_files[cnf_inst])
    print('n,m', n, m)
    Matrix_cnf = np.zeros((m, n), dtype=np.int)
    lencnf = np.zeros(m, dtype=np.int)
    for i in range(m):
        cnf = cnfs[i]
        lencnf[i] = len(cnf)
        for j in range(len(cnf)):
            k = abs(cnf[j])
            s = k / cnf[j]
            Matrix_cnf[i][k - 1] = s
    s1 = 7
    MATRIX_SIZE = n
    Matrix_cnf = Matrix_cnf * s1
    lencnf = lencnf * s1
    Thresholds = -lencnf + s1
    print("start main")
    startTime = time.perf_counter()
    best_candidate = runGSRWAlgorithm(abs_cnfs_1, m, Matrix_cnf, Thresholds, MATRIX_SIZE, NITER)
    endTime = time.perf_counter()
    d = endTime - startTime
    print("time for total:")
    print(f"total time: {d}")
    print(f"best energy is: {best_candidate.energy}")
    print(f"best vector is ")
    S_best = best_candidate.matrix.T
    print(S_best)
    unsatclauses, num_satclauses, T = verif_3sat_v2(cnfs, S_best)
    print('num of sat clauses', num_satclauses)
    print("Done")

if __name__ == "__main__":
    main()
