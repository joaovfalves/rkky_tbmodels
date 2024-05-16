#imports the common.py functions for the calculation of the Green's functions
from common import *

#calculates the Greens function one epslion at a time, necessary for the paralelization sript

def compute_gi_2band_for_eps(eps, rvec, r, lambd, Nq, tx=0.5, ty=0.5, tz=0.5, m=1,q0=np.pi/2):
    temp_g0, temp_gx, temp_gy, temp_gz = (np.zeros(len(rvec), dtype=complex) for _ in range(4))
    for j, rj in enumerate(rvec):
        r_vec = rj * r
        g = Gi_2band(r_vec, eps, lambd, Nq, tx,ty,tz,m,q0)
        temp_g0[j], temp_gx[j], temp_gy[j], temp_gz[j] = g
    return temp_g0, temp_gx, temp_gy, temp_gz

#calculates the matrix corresponding to the Green's function for the grid eps x r with ranges defined
#stores this matrix in a txt file 
def calculate_2d_2band_epsxr(lambd, axis = "x",epsmin = -10,epsmax=0, Neps = 301, rmax = 10, Nr = 301, Nq = 101, tx=0.5, ty=0.5, tz=0.5, m=1, q0=np.pi/2):
    epsvec = np.linspace(epsmin, epsmax, Neps)
    rvec = np.linspace(-rmax, rmax, Nr)
    # ... (rest of the function setup)
    direction_vectors = {"x": np.array((1, 0, 0)), "y": np.array((0, 1, 0)), "z": np.array((0, 0, 1))}
    r = direction_vectors.get(axis, np.array((1, 0, 0)))

    g0 = np.zeros((Nr, Neps), dtype=complex)
    gx = np.zeros((Nr, Neps), dtype=complex)
    gy = np.zeros((Nr, Neps), dtype=complex)
    gz = np.zeros((Nr, Neps), dtype=complex)

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(compute_gi_2band_for_eps, eps, rvec, r, lambd, Nq, tx, ty, tz, m,q0): i for i, eps in enumerate(epsvec)}
        for future in as_completed(futures):
            i = futures[future]
            result = future.result()
            g0[:, i], gx[:, i], gy[:, i], gz[:, i] = result
            progress = (i + 1) / len(epsvec) * 100
            clear_output(wait=True)
            print(f"Progress: {progress:.2f}%")

    # Save the results in a single loop
    file_prefix = f"{axis}_lamb={lambd}_rmax={rmax}_Nr={Nr}_Neps={Neps}_Nq={Nq}_tx={tx}_ty={ty}_tz={tz}_m={m}_k0={q0}_epslimits={epsmin}~{epsmax}"
    for g, suffix in zip([g0, gx, gy, gz], ['s0', 'sx', 'sy', 'sz']):
        filename = f'txt/[{suffix}][2band]{file_prefix}.txt'
        np.savetxt(filename, g)

def main():    
    
    #calculate_2d_2band_epsxr(lambd=0.01,axis = "x",epsmin = -5,epsmax=-4, Neps = 100, rmax = 5, Nr = 500, Nq = 100, tx=0.5, ty=0.5, tz=0.5, m=1, q0=2*np.pi/4)
    #calculate_2d_2band_epsxr(lambd=0.01,axis = "x",epsmin = -4,epsmax=-3, Neps = 100, rmax = 5, Nr = 500, Nq = 100, tx=0.5, ty=0.5, tz=0.5, m=1, q0=0*np.pi/4)
    calculate_2d_2band_epsxr(lambd=0.01,axis = "x",epsmin = -3,epsmax=-2, Neps = 100, rmax = 5, Nr = 500, Nq = 100, tx=0.5, ty=0.5, tz=0.5, m=1, q0=0*np.pi/4)
    #calculate_2d_2band_epsxr(lambd=0.01,axis = "x",epsmin = -2,epsmax=-1, Neps = 100, rmax = 5, Nr = 500, Nq = 100, tx=0.5, ty=0.5, tz=0.5, m=1, q0=0*np.pi/4)
    #calculate_2d_2band_epsxr(lambd=0.01,axis = "x",epsmin = -1,epsmax=0, Neps = 100, rmax = 5, Nr = 500, Nq = 100, tx=0.5, ty=0.5, tz=0.5, m=1, q0=0*np.pi/4)
        
if __name__ == '__main__':
    main()