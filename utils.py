import numpy as np
import re

def read_cnf(cnf_file):
    # 3-SAT cnf file `a.cnf` format:
    # 1. file can start with comments, that is lines begining with the character c
    # 2. Right after the comments, there is the line p cnf nbvar nbclauses indicating that the instance is in CNF format; i
    #   nbvar is the exact number of variables appearing in the file; nbclauses is the exact number of clauses contained in the file.
    # 3. Then the clauses follow.
    #   Each clause is a sequence of distinct non-null numbers between -nbvar and nbvar ending with 0 on the same line;
    #   it cannot contain the opposite literals i and -i simultaneously. Positive numbers denote the corresponding variables.
    #   Negative numbers denote the negations of the corresponding variables.
    # for example:
    # c
    # c start with comments
    # c
    # p cnf 5 3
    # 1 -5 4 0
    # -1 5 3 4 0
    # -3 -4 0
    # details: http://www.satcompetition.org/2009/format-benchmarks2009.html
    cnfs = []
    n = m = 0
    with open(cnf_file) as f:
        print(f"[READ] {cnf_file}")
        for line in f.readlines():
            line = line.strip(' \n')
            if line.startswith('c') or line == '0' or line == '%' or line == '\r\n' or line == '\n':
                continue
            if line.strip() == "":
                continue
            if line.startswith('p'):
                info = re.sub(' +', ' ', line).split(' ')
                print('info',info)
                assert len(info) == 4
                n,m = int(info[-2]), int(info[-1])
                print(f'[READ] {n} variables, {m} clauses')
                continue
            #if len(line) < 7:
                #continue
            else:
                num_list = line.split()[:-1]
                #num_list = line.split('\t')[:-1]
                #print('num_list',num_list)
                clause = [int(n) for n in num_list]
                #print('clause',clause)
                cnfs.append(clause)
            #print('cnfs',cnfs)
        print(f"[READ] {len(cnfs)} {len(clause)}-SAT data loaded")
    return cnfs, n, m

def read_cnf_GSRW(cnf_file):
    # 3-SAT cnf file `a.cnf` format:
    # 1. file can start with comments, that is lines begining with the character c
    # 2. Right after the comments, there is the line p cnf nbvar nbclauses indicating that the instance is in CNF format; i
    #   nbvar is the exact number of variables appearing in the file; nbclauses is the exact number of clauses contained in the file.
    # 3. Then the clauses follow.
    #   Each clause is a sequence of distinct non-null numbers between -nbvar and nbvar ending with 0 on the same line;
    #   it cannot contain the opposite literals i and -i simultaneously. Positive numbers denote the corresponding variables.
    #   Negative numbers denote the negations of the corresponding variables.
    # for example:
    # c
    # c start with comments
    # c
    # p cnf 5 3
    # 1 -5 4 0
    # -1 5 3 4 0
    # -3 -4 0
    # details: http://www.satcompetition.org/2009/format-benchmarks2009.html
    cnfs = []
    abs_cnfs_1 =[]
    n = m = 0
    with open(cnf_file) as f:
        print(f"[READ] {cnf_file}")
        for line in f.readlines():
            line = line.strip(' \n')
            if line.startswith('c') or line == '0' or line == '%' or line == '\r\n' or line == '\n':
                continue
            if line.strip() == "":
                continue
            if line.startswith('p'):
                info = re.sub(' +', ' ', line).split(' ')
                print('info',info)
                assert len(info) == 4
                n,m = int(info[-2]), int(info[-1])
                print(f'[READ] {n} variables, {m} clauses')
                continue
            #if len(line) < 7:
                #continue
            else:
                num_list = line.split()[:-1]
                #num_list = line.split('\t')[:-1]
                #print('num_list',num_list)
                clause = [int(n) for n in num_list]
                absclause = [abs(int(n))-1 for n in num_list]
                #print('clause',clause)
                #print('absclause',absclause)
                cnfs.append(clause)
                abs_cnfs_1.append(absclause)
            #print('cnfs',cnfs)
        print(f"[READ] {len(cnfs)} {len(clause)}-SAT data loaded")
    return cnfs, abs_cnfs_1, n, m

def Three_sat_toQUBO_v2(cnf_list, n, m):
    """
    inputs:
    - cnf_list: a list of clauses in conjunctive normal form, `len(cnf_list)==m`
    - n: the number of original variables
    - m: the number of clauses,
    outputs:
    - Q: QUBO matrix
    - n: the number of original variables
    - m: the number of clauses
    - K: the constant of the QUBO model
    """
    assert len(cnf_list) == m
    n_vars = n + m
    Q = np.zeros((n_vars, n_vars))
    K = 0 # constant
    count = 0
    #print("[INFO] Start qubo transformation...")
    for i,cnf in enumerate(cnf_list):
        lcnf = len(cnf)
        # vi: var index; s: value
        vi = [abs(v) for v in cnf]
        #print('vi',vi)
        s = [0 if v<0 else 1 for v in cnf]
        #print('s',s)
        vi,s = zip(*sorted(zip(vi, s)))
        #print('vi', vi[0])
        #print('s', s[0])
        if lcnf == 2:
            v1, v2 = vi
            s1, s2 = s
        else:
            v1,v2,v3 = vi
            s1,s2,s3 = s
        # update Q and K
        if lcnf == 2:
            if s1 == 1 and s2 == 1:
                K += 0
                Q[v1 - 1, v1 - 1] += 1
                Q[v2 - 1, v2 - 1] += 1
                Q[v1 - 1, v2 - 1] += -1
            elif s1 == 0 and s2 == 1:
                K += 1
                Q[v1 - 1, v1 - 1] += -1
                Q[v2 - 1, v2 - 1] += 0
                Q[v1 - 1, v2 - 1] += 1
            elif s1 == 1 and s2 == 0:
                K += 1
                Q[v1 - 1, v1 - 1] += 0
                Q[v2 - 1, v2 - 1] += -1
                Q[v1 - 1, v2 - 1] += 1
            else:
                K += 1
                Q[v1 - 1, v1 - 1] += 0
                Q[v2 - 1, v2 - 1] += 0
                Q[v1 - 1, v2 - 1] += -1
        if lcnf == 3:
            if s1==1 and s2==1 and s3==1:
                K += 0
                Q[n+count, n+count] += -2
                Q[v1-1,v1-1] += 1
                Q[v2-1,v2-1] += 1
                Q[v3-1,v3-1] += 1
                Q[v1-1,v2-1] += -1
                Q[v1-1,v3-1] += -1
                Q[v2-1,v3-1] += -1
                Q[v1-1, n+count] += 1
                Q[v2-1, n+count] += 1
                Q[v3-1, n+count] += 1
            elif s1==0 and s2==1 and s3==1:
                K += 1
                Q[n+count, n+count] += -1
                Q[v1-1,v1-1] += -1
                Q[v2-1,v2-1] += 0
                Q[v3-1,v3-1] += 0
                Q[v1-1,v2-1] += 1
                Q[v1-1,v3-1] += 1
                Q[v2-1,v3-1] += -1
                Q[v1-1, n+count] += -1
                Q[v2-1, n+count] += 1
                Q[v3-1, n+count] += 1
            elif s1==1 and s2==0 and s3==1:
                K += 1
                Q[n+count, n+count] += -1
                Q[v1-1,v1-1] += 0
                Q[v2-1,v2-1] += -1
                Q[v3-1,v3-1] += 0
                Q[v1-1,v2-1] += 1
                Q[v1-1,v3-1] += -1
                Q[v2-1,v3-1] += 1
                Q[v1-1, n+count] += 1
                Q[v2-1, n+count] += -1
                Q[v3-1, n+count] += 1
            elif s1==1 and s2==1 and s3==0:
                K += 1
                Q[n+count, n+count] += -1
                Q[v1-1,v1-1] += 0
                Q[v2-1,v2-1] += 0
                Q[v3-1,v3-1] += -1
                Q[v1-1,v2-1] += -1
                Q[v1-1,v3-1] += 1
                Q[v2-1,v3-1] += 1
                Q[v1-1, n+count] += 1
                Q[v2-1, n+count] += 1
                Q[v3-1, n+count] += -1
            elif s1==0 and s2==0 and s3==1:
                K += 1
                Q[n+count, n+count] += 0
                Q[v1-1,v1-1] += 0
                Q[v2-1,v2-1] += 0
                Q[v3-1,v3-1] += -1
                Q[v1-1,v2-1] += -1
                Q[v1-1,v3-1] += 1
                Q[v2-1,v3-1] += 1
                Q[v1-1, n+count] += -1
                Q[v2-1, n+count] += -1
                Q[v3-1, n+count] += 1
            elif s1==0 and s2==1 and s3==0:
                K += 1
                Q[n+count, n+count] += 0
                Q[v1-1,v1-1] += 0
                Q[v2-1,v2-1] += -1
                Q[v3-1,v3-1] += 0
                Q[v1-1,v2-1] += 1
                Q[v1-1,v3-1] += -1
                Q[v2-1,v3-1] += 1
                Q[v1-1, n+count] += -1
                Q[v2-1, n+count] += 1
                Q[v3-1, n+count] += -1
            elif s1==1 and s2==0 and s3==0:
                K += 1
                Q[n+count, n+count] += 0
                Q[v1-1,v1-1] += -1
                Q[v2-1,v2-1] += 0
                Q[v3-1,v3-1] += 0
                Q[v1-1,v2-1] += 1
                Q[v1-1,v3-1] += 1
                Q[v2-1,v3-1] += -1
                Q[v1-1, n+count] += 1
                Q[v2-1, n+count] += -1
                Q[v3-1, n+count] += -1
            else:
                K += 0
                Q[n+count, n+count] += 1
                Q[v1-1,v1-1] += 1
                Q[v2-1,v2-1] += 1
                Q[v3-1,v3-1] += 1
                Q[v1-1,v2-1] += -1
                Q[v1-1,v3-1] += -1
                Q[v2-1,v3-1] += -1
                Q[v1-1, n+count] += -1
                Q[v2-1, n+count] += -1
                Q[v3-1, n+count] += -1
            count += 1

    Q=-Q # From max to min
    Q=(Q+Q.T)/2
    return Q,n,m,K

def getKL(Q,MATRIX_SIZE):
    # This function get the matrix K and external field L
    # Calculate K matrix for Q Matrix
    K = Q.copy()
    K = -0.5 * K
    for i in range(MATRIX_SIZE):
        K[i][i] = 0
    L = np.zeros((MATRIX_SIZE, 1))  # external magnetic field
    L = np.sum(Q, axis=1)
    L = -0.5 * L

    return K, L

def quad_K(K):
    Bit = 4
    SL = -2**(Bit-1)
    SR = 2**(Bit-1)-1
    KL = K.min()
    KR = K.max()
    epsilon = 1E-6
    if KR<0:
        scale = SL/KL
    elif KL>0:
        scale = SR/KR
    else:
        scale = min(SL/(KL+epsilon),SR/(KR+epsilon))
    KS = np.around(K*scale)

    return scale, KS
def verif_sou(cnf_list, S_sou, m):
    """
    This function verify the solution of the 3-SAT problem by calculation the energy  or the number of the satisfied clauses
    inputs:
    - S_sou: the soultion of the 3-SAT problem, `len(S_sou)==n`
    - n: the number of the variables of the 3-SAT problem
    - m: the number of clauses,
    - w: extra variable wi
    outputs:
    - eng: the energy of the 3-SAT problem correspond to the solution S_sou
    """
    assert len(cnf_list) == m
    #assert len(S_sou) == n
    eng = 0
    uncla = []
    for i,cnf in enumerate(cnf_list):
        lcnf = len(cnf)
        # vi: var index; s: value
        vi = [abs(v) for v in cnf]
        #print('vi',vi)
        s = [0 if v<0 else 1 for v in cnf]
        #print('cnf', cnf)
        #print('s',s)
        vi,s = zip(*sorted(zip(vi, s)))
        #print('vi', vi)
        #print('s', s)
        x = 0
        if lcnf == 2:
            v1, v2 = vi
            s1, s2 = s
            a = s1 * S_sou[v1 - 1] + (1 - s1) * (1 - S_sou[v1 - 1])
            b = s2 * S_sou[v2 - 1] + (1 - s2) * (1 - S_sou[v2 - 1])
            temp = a + b -a*b
            eng += temp
            if temp == 0:
                uncla.append(i)
                #print('i', i)
                #print(v1, s1, S_sou[v1 - 1])
                #print(v2, s2, S_sou[v2 - 1])
        elif lcnf == 3:
            v1, v2, v3 = vi
            s1, s2, s3 = s
            # calculation the energy
            a = s1 * S_sou[v1 - 1] + (1 - s1) * (1 - S_sou[v1 - 1])
            b = s2 * S_sou[v2 - 1] + (1 - s2) * (1 - S_sou[v2 - 1])
            c = s3 * S_sou[v3 - 1] + (1 - s3) * (1 - S_sou[v3 - 1])
            temp = a + b + c - a * b - a * c - b * c + a * b * c
            eng += temp
            if temp == 0:
                uncla.append(i)
                #print('i', i)
                #print(v1, s1, S_sou[v1 - 1])
                #print(v2, s2, S_sou[v2 - 1])
                #print(v3, s3, S_sou[v3 - 1])
        else:
            for j in range(lcnf):
                vj = vi[j]
                sj = s[j]
                x = sj * S_sou[vj - 1] + (1 - sj) * (1 - S_sou[vj - 1])
                if x == 1:
                    eng += 1
                    break
            #if x == 0:
                #print('i', i)
    return eng

def verif_3sat(cnf_list, S_sou):
    """
    This function verify the solution of the 3-SAT problem by calculation the energy  or the number of the satisfied clauses
    inputs:
    - S_sou: the soultion of the 3-SAT problem, `len(S_sou)==n`
    - n: the number of the variables of the 3-SAT problem
    - m: the number of clauses,
    outputs:
    - eng: the energy of the 3-SAT problem correspond to the solution S_sou
    """
    eng = 0
    for cnf in cnf_list:
        for v in cnf:
            if v>0:
                x = S_sou[v - 1]
            else:
                x = 1 - S_sou[-v - 1]
            if x == 1:
                eng += 1
                break
    return eng

def verif_3sat_v2(cnf_list, Spin):
    """
    This function verify the solution of the 3-SAT problem by calculation the energy  or the number of the satisfied clauses
    inputs:
    - S_sou: the soultion of the 3-SAT problem, `len(S_sou)==n`
    - n: the number of the variables of the 3-SAT problem
    - m: the number of clauses,
    outputs:
    - eng: the energy of the 3-SAT problem correspond to the solution S_sou
    """
    eng = 0
    T = 0
    all_clause = np.arange(len(cnf_list))
    for i, cnf in enumerate(cnf_list):
        for v in cnf:
            if v>0:
                x = Spin[v - 1]
            else:
                x = - Spin[-v - 1]
            if x == 1:
                eng += 1
                np.delete(all_clause,i)
                break
    if eng == len(cnf_list):
        T = 1
    return all_clause, eng, T