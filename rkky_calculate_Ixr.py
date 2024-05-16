import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
import sys
from IPython.display import clear_output

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

from mpl_toolkits.axisartist.axislines import AxesZero

plt.rc('text', usetex=True)
plt.rc('font',family='serif')


sigma_0 = np.array([[1, 0], [0, 1]])
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

#calculate the exchange integral terms via the green's functions in real space as a function of the postion of the impurities r
def calculate_1d_r(name, J):
    # nome do arquivo tem a forma
    #[sz][minimal]x_lamb=0.01_rmax=5_Nr=500_Neps=100_Nq=100_tx=0.5_ty=0.5_tz=0.5_m=1_k0=1.5707963267948966_epslimits=-1.0~0.0


    li = np.char.find(name,"lamb=")
    lf = np.char.find(name,"_rmax=")
    lamb = name[li+5:lf]

    rf = np.char.find(name,"_Nr=")
    r = name[lf+6:rf]

    Nrf = np.char.find(name,"_Neps=")
    Nr = name[rf+4:Nrf]

    Nepsf = np.char.find(name,"_Nq=")
    Neps = name[Nrf+6:Nepsf]

    k0f = np.char.find(name,"_epslimits=")
    eps0f = np.char.find(name,"~")

    eps0 = name[k0f + 11:eps0f]
    epsf = name[eps0f+1:]


    if name[5:11] == "linear":

        axis = name[12]


        Nqf = np.char.find(name,"_v0=")
        Nq = name[Nepsf+4:Nqf]

        v0f = np.char.find(name,"_Q=")
        v0 = name[Nqf+4:v0f]

        Q = name[v0f+3:]


        figname = "[linear]-2d-"+name[4:]



    if name[5:11] == "minima":

        axis = name[13]


        Nqf = np.char.find(name,"_tx=")
        Nq = name[Nepsf+4:Nqf]

        txf = np.char.find(name,"_ty=")
        tx = name[Nqf+4:txf]

        tyf = np.char.find(name,"_tz=")
        ty = name[txf+4:tyf]

        tzf = np.char.find(name,"_m=")
        tz = name[tyf+4:tzf]

        mf = np.char.find(name,"_k0=")
        m = name[tzf+3:mf]

        k0 = name[mf+4:k0f]

        figname = "[minimal]-2d-"+name[4:]



    Nr = int(Nr)
    Neps = int(Neps)

    k0f = np.char.find(name,"_epslimits=")
    eps0f = np.char.find(name,"~")


    eps0 = name[k0f + 11:eps0f]
    epsf = name[eps0f+1:]


    plt.rc('font',size= round(int(Nr)/5))

    epsvec = np.linspace(float(eps0),float(epsf), int(Neps))

    g0p = np.zeros((int(Nr/2),int(Neps)) , dtype = complex)
    g0m = np.copy(g0p)

    gxp = np.copy(g0p)
    gxm = np.copy(g0p)

    gyp = np.copy(g0p)
    gym = np.copy(g0p)

    gzp = np.copy(g0p)
    gzm = np.copy(g0p)

#separates the plus r and the minus r part of the green's function

    g0p[:,:] = np.loadtxt("drive/MyDrive/txt/greens/[s0]"+name[4:]+".txt", dtype=complex)[int(Nr/2):,:]
    gxp[:,:] = np.loadtxt("drive/MyDrive/txt/greens/[sx]"+name[4:]+".txt", dtype=complex)[int(Nr/2):,:]
    gyp[:,:] = np.loadtxt("drive/MyDrive/txt/greens/[sy]"+name[4:]+".txt", dtype=complex)[int(Nr/2):,:]
    gzp[:,:] = np.loadtxt("drive/MyDrive/txt/greens/[sz]"+name[4:]+".txt", dtype=complex)[int(Nr/2):,:]

#the flip is for relating r to -r
#i.e. g0p[i,epsj] = G(+r_i,epsj), g0m[i,:] = G(-r_i,epsj)
    g0m[:,:] = np.flip( np.loadtxt("drive/MyDrive/txt/greens/[s0]"+name[4:]+".txt", dtype=complex)[:int(Nr/2),:], axis =0)
    gxm[:,:] = np.flip( np.loadtxt("drive/MyDrive/txt/greens/[sx]"+name[4:]+".txt", dtype=complex)[:int(Nr/2),:], axis =0)
    gym[:,:] = np.flip( np.loadtxt("drive/MyDrive/txt/greens/[sy]"+name[4:]+".txt", dtype=complex)[:int(Nr/2),:], axis =0)
    gzm[:,:] = np.flip( np.loadtxt("drive/MyDrive/txt/greens/[sz]"+name[4:]+".txt", dtype=complex)[:int(Nr/2),:], axis =0)



    IH = np.zeros((int(Nr), int(Neps)), dtype=complex)

    IIsx = np.copy(IH)
    IIsy = np.copy(IH)
    IIsz = np.copy(IH)

    IDMx = np.copy(IH)
    IDMy = np.copy(IH)
    IDMz = np.copy(IH)

    Ifrxy = np.copy(IH)
    Ifrxz = np.copy(IH)
    Ifryz = np.copy(IH)


    # the integral is over negative epsilons
    deps = np.abs(epsvec[0] - epsvec[1])

    A=- 2/(np.pi) * J**2
    IH = np.trapz( np.imag( A * g0p * g0m), epsvec, deps, axis=-1)

    IIsx = np.trapz( np.imag( A* ( gxp*gxm -  gyp*gym - gzp*gzm - gxp*gym - gyp*gxm- gxp*gzm - gzp*gxm) ), epsvec, deps, axis=-1)
    IIsy = np.trapz( np.imag( A* ( gyp*gym -  gxp*gxm - gzp*gzm - gxp*gym - gyp*gxm- gyp*gzm - gzp*gym) ), epsvec, deps, axis=-1)
    IIsz = np.trapz( np.imag( A* ( gzp*gzm -  gxp*gxm - gyp*gym - gxp*gzm - gzp*gxm- gzp*gym - gyp*gzm) ), epsvec, deps, axis=-1)

    IDMx = np.trapz( np.imag( A * ( g0p*gxm - gxp*g0m ) ), epsvec, deps, axis=-1)
    IDMy = np.trapz( np.imag(A * ( g0p*gym - gyp*g0m ) ), epsvec, deps, axis=-1)
    IDMz = np.trapz( np.imag( A * ( g0p*gzm - gzp*g0m ) ), epsvec, deps, axis=-1)

    Ifrxy = np.trapz( np.imag( A * ( gxp * gym + gyp * gxm ) ), epsvec, deps, axis=-1)
    Ifrxz = np.trapz( np.imag( A * ( gxp * gzm + gzp * gxm ) ), epsvec, deps, axis=-1)
    Ifryz = np.trapz( np.imag( A * ( gyp * gzm + gzp * gym ) ), epsvec, deps, axis=-1)

    np.savetxt('drive/MyDrive/txt/int/[IH]'+name[4:]+'.txt', IH)

    np.savetxt('drive/MyDrive/txt/int/[Isx]'+name[4:]+'.txt', IIsx)
    np.savetxt('drive/MyDrive/txt/int/[Isy]'+name[4:]+'.txt', IIsy)
    np.savetxt('drive/MyDrive/txt/int/[Isz]'+name[4:]+'.txt', IIsz)

    np.savetxt('drive/MyDrive/txt/int/[IDMx]'+name[4:]+'.txt', IDMx)
    np.savetxt('drive/MyDrive/txt/int/[IDMy]'+name[4:]+'.txt', IDMy)
    np.savetxt('drive/MyDrive/txt/int/[IDMz]'+name[4:]+'.txt', IDMz)

    np.savetxt('drive/MyDrive/txt/int/[Ifrxy]'+name[4:]+'.txt', Ifrxy)
    np.savetxt('drive/MyDrive/txt/int/[Ifrxz]'+name[4:]+'.txt', Ifrxz)
    np.savetxt('drive/MyDrive/txt/int/[Ifryz]'+name[4:]+'.txt', Ifryz)

#calculate_1d_r("[sx][minimal]x_lamb=0.01_rmax=5_Nr=500_Neps=400_Nq=100_tx=0.5_ty=0.5_tz=0.5_m=1_k0=0.0_epslimits=-4~0", J=1)
#0.0  0.7853981633974483   1.5707963267948966   2.356194490192345   3.141592653589793

#calculate_1d_r("[s0][minimal]x_lamb=0.01_rmax=5_Nr=500_Neps=100_Nq=100_tx=0.5_ty=0.5_tz=0.5_m=1_k0=0.0_epslimits=-1~0", J=1)
#calculate_1d_r("[s0][minimal]x_lamb=0.01_rmax=5_Nr=500_Neps=400_Nq=100_tx=0.5_ty=0.5_tz=0.5_m=1_k0=0.0_epslimits=-4~0", J=1)
#calculate_1d_r("[s0][minimal]z_lamb=0.01_rmax=5_Nr=500_Neps=100_Nq=100_tx=0.5_ty=0.5_tz=0.5_m=1_k0=0.0_epslimits=-1~0", J=1)
#calculate_1d_r("[s0][minimal]z_lamb=0.01_rmax=5_Nr=500_Neps=400_Nq=100_tx=0.5_ty=0.5_tz=0.5_m=1_k0=0.0_epslimits=-4~0", J=1)
#calculate_1d_r("[s0][minimal]xy_lamb=0.01_rmax=5_Nr=500_Neps=100_Nq=100_tx=0.5_ty=0.5_tz=0.5_m=1_k0=0.0_epslimits=-1~0", J=1)
#calculate_1d_r("[s0][minimal]xy_lamb=0.01_rmax=5_Nr=500_Neps=500_Nq=100_tx=0.5_ty=0.5_tz=0.5_m=1_k0=0.0_epslimits=-5~0", J=1)


#calculate_1d_r("[s0][minimal]x_lamb=0.01_rmax=5_Nr=500_Neps=100_Nq=100_tx=0.5_ty=0.5_tz=0.5_m=1_k0=0.7853981633974483_epslimits=-1.0~0.0", J=1)
#calculate_1d_r("[s0][minimal]x_lamb=0.01_rmax=5_Nr=500_Neps=500_Nq=100_tx=0.5_ty=0.5_tz=0.5_m=1_k0=0.7853981633974483_epslimits=-5~0", J=1)
#calculate_1d_r("[s0][minimal]z_lamb=0.01_rmax=5_Nr=500_Neps=100_Nq=100_tx=0.5_ty=0.5_tz=0.5_m=1_k0=0.7853981633974483_epslimits=-1.0~0.0", J=1)
#calculate_1d_r("[s0][minimal]z_lamb=0.01_rmax=5_Nr=500_Neps=500_Nq=100_tx=0.5_ty=0.5_tz=0.5_m=1_k0=0.7853981633974483_epslimits=-5~0", J=1)
#calculate_1d_r("[s0][minimal]xy_lamb=0.01_rmax=5_Nr=500_Neps=100_Nq=100_tx=0.5_ty=0.5_tz=0.5_m=1_k0=0.7853981633974483_epslimits=-1~0", J=1)
#calculate_1d_r("[s0][minimal]xy_lamb=0.01_rmax=5_Nr=500_Neps=500_Nq=100_tx=0.5_ty=0.5_tz=0.5_m=1_k0=0.7853981633974483_epslimits=-5~0", J=1)



#calculate_1d_r("[s0][minimal]x_lamb=0.01_rmax=5_Nr=500_Neps=100_Nq=100_tx=0.5_ty=0.5_tz=0.5_m=1_k0=1.5707963267948966_epslimits=-1~0", J=1)
#calculate_1d_r("[s0][minimal]x_lamb=0.01_rmax=5_Nr=500_Neps=500_Nq=100_tx=0.5_ty=0.5_tz=0.5_m=1_k0=1.5707963267948966_epslimits=-5~0", J=1)
#calculate_1d_r("[s0][minimal]z_lamb=0.01_rmax=5_Nr=500_Neps=100_Nq=100_tx=0.5_ty=0.5_tz=0.5_m=1_k0=1.5707963267948966_epslimits=-1~0", J=1)
#calculate_1d_r("[s0][minimal]z_lamb=0.01_rmax=5_Nr=500_Neps=500_Nq=100_tx=0.5_ty=0.5_tz=0.5_m=1_k0=1.5707963267948966_epslimits=-5~0", J=1)
#calculate_1d_r("[s0][minimal]xy_lamb=0.01_rmax=5_Nr=500_Neps=100_Nq=100_tx=0.5_ty=0.5_tz=0.5_m=1_k0=1.5707963267948966_epslimits=-1~0", J=1)
#calculate_1d_r("[s0][minimal]xy_lamb=0.01_rmax=5_Nr=500_Neps=500_Nq=100_tx=0.5_ty=0.5_tz=0.5_m=1_k0=1.5707963267948966_epslimits=-5~0", J=1)


#calculate_1d_r("[s0][minimal]x_lamb=0.01_rmax=5_Nr=500_Neps=100_Nq=100_tx=0.5_ty=0.5_tz=0.5_m=1_k0=2.356194490192345_epslimits=-1.0~0.0", J=1)
#calculate_1d_r("[s0][minimal]x_lamb=0.01_rmax=5_Nr=500_Neps=500_Nq=100_tx=0.5_ty=0.5_tz=0.5_m=1_k0=2.356194490192345_epslimits=-5~0", J=1)
#calculate_1d_r("[s0][minimal]z_lamb=0.01_rmax=5_Nr=500_Neps=100_Nq=100_tx=0.5_ty=0.5_tz=0.5_m=1_k0=2.356194490192345_epslimits=-1.0~0.0", J=1)
#calculate_1d_r("[s0][minimal]z_lamb=0.01_rmax=5_Nr=500_Neps=500_Nq=100_tx=0.5_ty=0.5_tz=0.5_m=1_k0=2.356194490192345_epslimits=-5~0", J=1)
#calculate_1d_r("[s0][minimal]xy_lamb=0.01_rmax=5_Nr=500_Neps=100_Nq=100_tx=0.5_ty=0.5_tz=0.5_m=1_k0=2.356194490192345_epslimits=-1~0", J=1)
#calculate_1d_r("[s0][minimal]xy_lamb=0.01_rmax=5_Nr=500_Neps=500_Nq=100_tx=0.5_ty=0.5_tz=0.5_m=1_k0=2.356194490192345_epslimits=-5~0", J=1)


#calculate_1d_r("[s0][minimal]x_lamb=0.01_rmax=5_Nr=500_Neps=100_Nq=100_tx=0.5_ty=0.5_tz=0.5_m=1_k0=3.141592653589793_epslimits=-1.0~0.0", J=1)
#calculate_1d_r("[s0][minimal]x_lamb=0.01_rmax=5_Nr=500_Neps=500_Nq=100_tx=0.5_ty=0.5_tz=0.5_m=1_k0=3.141592653589793_epslimits=-5~0", J=1)
#calculate_1d_r("[s0][minimal]z_lamb=0.01_rmax=5_Nr=500_Neps=100_Nq=100_tx=0.5_ty=0.5_tz=0.5_m=1_k0=3.141592653589793_epslimits=-1.0~0.0", J=1)
#calculate_1d_r("[s0][minimal]z_lamb=0.01_rmax=5_Nr=500_Neps=500_Nq=100_tx=0.5_ty=0.5_tz=0.5_m=1_k0=3.141592653589793_epslimits=-5~0", J=1)
#calculate_1d_r("[s0][minimal]xy_lamb=0.01_rmax=5_Nr=500_Neps=100_Nq=100_tx=0.5_ty=0.5_tz=0.5_m=1_k0=3.141592653589793_epslimits=-1~0", J=1)
#calculate_1d_r("[s0][minimal]xy_lamb=0.01_rmax=5_Nr=500_Neps=500_Nq=100_tx=0.5_ty=0.5_tz=0.5_m=1_k0=3.141592653589793_epslimits=-5~0", J=1)