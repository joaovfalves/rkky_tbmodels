#imports the common.py functions for the calculation of the Green's functions
from common import *

#calculates the Greens function one epslion at a time, necessary for the paralelization sript

def compute_gi_2band_for_eps(eps, q0vec, r, lambd, Nq, tx=0.5, ty=0.5, tz=0.5, m=1):
    temp_g0, temp_gx, temp_gy, temp_gz = (np.zeros(len(q0vec), dtype=complex) for _ in range(4))
    for j, q0j in enumerate(q0vec):
        g = Gi_2band(r, eps, lambd, Nq, tx,ty,tz,m,q0j)
        temp_g0[j], temp_gx[j], temp_gy[j], temp_gz[j] = g
    return temp_g0, temp_gx, temp_gy, temp_gz


#calculates the matrix corresponding to the Green's function for the grid eps x q0 with ranges defined
#stores this matrix in a txt file 

def calculate_2d_2band_epsxq0(lambd,epsmin = -10,epsmax=0, Neps = 301, r= (0,0,0), Nq0 = 301, Nq = 101, tx=0.5, ty=0.5, tz=0.5, m=1):
    epsvec = np.linspace(epsmin, epsmax, Neps)
    q0vec = np.linspace(0, np.pi, Nq0)


    g0 = np.zeros((Nq0, Neps), dtype=complex)
    gx = np.zeros((Nq0, Neps), dtype=complex)
    gy = np.zeros((Nq0, Neps), dtype=complex)
    gz = np.zeros((Nq0, Neps), dtype=complex)

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(compute_gi_2band_for_eps, eps, q0vec, r, lambd, Nq, tx, ty, tz, m): i for i, eps in enumerate(epsvec)}
        for future in as_completed(futures):
            i = futures[future]
            result = future.result()
            g0[:, i], gx[:, i], gy[:, i], gz[:, i] = result
            progress = (i + 1) / len(q0vec) * 100
            clear_output(wait=True)
            print(f"Progress: {progress:.2f}%")


    file_prefix = f"lamb={lambd}_r={r}_Nq0={Nq0}_Neps={Neps}_Nq={Nq}_tx={tx}_tx={ty}_tx={tz}_epslimits={epsmin}~{epsmax}_q0limits={q0vec[0]}~{q0vec[-1]}"
    for g, suffix in zip([g0, gx, gy, gz], ['s0', 'sx', 'sy', 'sz']):
        filename = f'txt/[{suffix}][2band]{file_prefix}.txt'
        np.savetxt(filename, g)
        

def main():    
    
    calculate_2d_2band_epsxq0(lambd=0.01,epsmin = -2,epsmax=-1, Neps = 150, r= (0,1,1), Nq0 = 150, Nq = 100, tx=0.5, ty=0.5, tz=0.5, m=1)
    calculate_2d_2band_epsxq0(lambd=0.01,epsmin = -2,epsmax=-1, Neps = 150, r= (0,-1,-1), Nq0 = 150, Nq = 100, tx=0.5, ty=0.5, tz=0.5, m=1)

if __name__ == '__main__':
    main()