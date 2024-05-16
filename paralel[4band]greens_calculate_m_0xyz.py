#imports the common.py functions for the calculation of the Green's functions
from common import *

#calculates the Greens function one epslion at a time, necessary for the paralelization sript
def compute_gi_4band_for_eps(eps, mvec, r, lambd, Nq, t):
    temp_g0, temp_gx, temp_gy, temp_gz = (np.zeros(len(mvec), dtype=complex) for _ in range(4))
    for j, mj in enumerate(mvec):
        g = Gi_4band(r, eps, lambd, Nq, t, mj)
        temp_g0[j], temp_gx[j], temp_gy[j], temp_gz[j] = g
    return temp_g0, temp_gx, temp_gy, temp_gz

#calculates the matrix corresponding to the Green's function for the grid eps x m with ranges defined
#stores this matrix in a txt file 
def calculate_2d_4band_epsxm(lambd,epsmin = -10,epsmax=0, Neps = 301, r= (0,0,0), Nm = 301, Nq = 101, t=1):
    epsvec = np.linspace(epsmin,epsmax, Neps)

    mvec = np.linspace(0, 1, Nm)

    g0, gx, gy, gz = (np.zeros((Nm, Neps), dtype=complex) for _ in range(4))
    

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(compute_gi_4band_for_eps, eps, mvec, r, lambd, Nq, t): i for i, eps in enumerate(epsvec)}
        for future in as_completed(futures):
            i = futures[future]
            result = future.result()
            g0[:, i], gx[:, i], gy[:, i], gz[:, i] = result
            progress = (i + 1) / len(epsvec) * 100
            clear_output(wait=True)
            print(f"Progress: {progress:.2f}%")

# Save the results in a loop
    file_prefix = f"lamb={lambd}_r={r}_Nm={Nm}_Neps={Neps}_Nq={Nq}_t={t}_mlimits={mvec[0]}~{mvec[-1]}_epslimits={epsmin}~{epsmax}"
    for g, suffix in zip([g0, gx, gy, gz], ['s0', 'sx', 'sy', 'sz']):
        filename = f'txt/[{suffix}][4band]{file_prefix}.txt'
        np.savetxt(filename, g)
        
        
def main():
    calculate_2d_4band_epsxm(lambd=0.01,epsmin = -1,epsmax=0, Neps = 150, r= (0, 1,0), Nm = 100, Nq = 100, t=1)
    
if __name__ == '__main__':
    main()