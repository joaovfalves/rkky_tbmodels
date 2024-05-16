import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
import sys
from IPython.display import clear_output

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

plt.rc('text', usetex=True)
plt.rc('font',family='serif')


sigma_0 = np.array([[1, 0], [0, 1]])
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

#calculate the exchange integral terms via the green's functions in real space as a function of the parameter q0
#it needs the function at +r and at -r
def calculate_1d_q(namep, namem, J):

    # nome do arquivo tem a forma
    #[sz][minimal]_lamb=0.01_r=(-1, 0, 0)_Nq0=150_Neps=150_Nq=100_tx=0.5_ty=0.5_tz=0.5_m=1_epslimits=-1.0~0.0_q0limits=0.0~3.141592653589793

    #precisamos de um arquivo para r e outro para -r !!!

    name = namep

    li = np.char.find(name,"lamb=")
    lf = np.char.find(name,"_r=")
    lamb = name[li+5:lf]

    Nepsf = np.char.find(name,"_Nq=")

    epsff = None

    if np.char.find(name, "linear") != np.array(-1):

        axis = name[12]

        rf = np.char.find(name,"_NQ=")
        r = name[lf+3:rf]

        NQf = np.char.find(name,"_Neps=")
        NQ = name[rf+4:NQf]

        Neps = name[NQf+6:Nepsf]

        Nqf = np.char.find(name,"_v0=")
        Nq = name[Nepsf+4:Nqf]

        Nq0 =NQ

        figname = "[linear]-2d-"+name[4:]

        epsff = np.char.find(name,"_q0limits=")



    if np.char.find(name, "minimal") != np.array(-1):

        axis = name[13]

        rf = np.char.find(name,"_Nq0=")
        r = name[lf+3:rf]

        Nq0f = np.char.find(name,"_Neps=")
        Nq0 = name[rf+5:Nq0f]

        Neps = name[Nq0f+6:Nepsf]

        Nqf = np.char.find(name,"_tx=")
        Nq = name[Nepsf+4:Nqf]

        txf = np.char.find(name,"_ty=")
        tx = name[Nqf+4:txf]

        tyf = np.char.find(name,"_tz=")
        ty = name[txf+4:tyf]

        tzf = np.char.find(name,"_m=")
        tz = name[tyf+4:tzf]

        #mf = np.char.find(name,"_k0=")
        m = name[tzf+3:]

        #k0 = name[mf+4:]

        epsff = np.char.find(name,"_q0limits=")

        figname = "[minimal]-2d-"+name[4:]

    k0f = np.char.find(name,"_epslimits=")
    eps0f = np.char.find(name,"~")


    eps0 = name[k0f + 11:eps0f]
    epsf = name[eps0f+1:epsff]
    #Nq0 = int(Nq0)
    #Neps = int(Neps)
    Nq0,Neps = np.loadtxt("drive/MyDrive/txt/greens/[s0]"+namep[4:]+".txt", dtype=complex).shape


    print(Nq0)


    plt.rc('font',size= round(int(Nq0)/5))

    epsvec = np.linspace(float(eps0),float(epsf), int(Neps))
    q0vec = np.linspace(-np.pi, np.pi, Nq0)

    g0p = np.zeros((int(Nq0),int(Neps)) , dtype = complex)
    g0m = np.copy(g0p)

    gxp = np.copy(g0p)
    gxm = np.copy(g0p)

    gyp = np.copy(g0p)
    gym = np.copy(g0p)

    gzp = np.copy(g0p)
    gzm = np.copy(g0p)


    g0p[:,:] = np.loadtxt("drive/MyDrive/txt/greens/[s0]"+namep[4:]+".txt", dtype=complex)
    gxp[:,:] = np.loadtxt("drive/MyDrive/txt/greens/[sx]"+namep[4:]+".txt", dtype=complex)
    gyp[:,:] = np.loadtxt("drive/MyDrive/txt/greens/[sy]"+namep[4:]+".txt", dtype=complex)
    gzp[:,:] = np.loadtxt("drive/MyDrive/txt/greens/[sz]"+namep[4:]+".txt", dtype=complex)


    g0m[:,:] = np.loadtxt("drive/MyDrive/txt/greens/[s0]"+namem[4:]+".txt", dtype=complex)
    gxm[:,:] = np.loadtxt("drive/MyDrive/txt/greens/[sx]"+namem[4:]+".txt", dtype=complex)
    gym[:,:] = np.loadtxt("drive/MyDrive/txt/greens/[sy]"+namem[4:]+".txt", dtype=complex)
    gzm[:,:] = np.loadtxt("drive/MyDrive/txt/greens/[sz]"+namem[4:]+".txt", dtype=complex)

    #q0vec_m = np.flip(q0vec[0:int(Nq0/2)])
    #q0vec_p = q0vec[int(Nq0/2):]

    IH = np.zeros((int(Nq0), int(Neps)), dtype=complex)

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


#calculate_1d_q("[sz][minimal]_lamb=0.01_r=(1, 0, 1)_Nq0=150_Neps=150_Nq=100_tx=0.5_ty=0.5_tz=0.5_m=1_epslimits=-1.0~0.0_q0limits=0.0~3.141592653589793", "[sz][minimal]_lamb=0.01_r=(-1, 0, -1)_Nq0=150_Neps=150_Nq=100_tx=0.5_ty=0.5_tz=0.5_m=1_epslimits=-1.0~0.0_q0limits=0.0~3.141592653589793", J=1)
#calculate_1d_q("[sz][minimal]_lamb=0.01_r=(1, 1, 0)_Nq0=150_Neps=150_Nq=100_tx=0.5_ty=0.5_tz=0.5_m=1_epslimits=-1.0~0.0_q0limits=0.0~3.141592653589793", "[sz][minimal]_lamb=0.01_r=(-1, -1, 0)_Nq0=150_Neps=150_Nq=100_tx=0.5_ty=0.5_tz=0.5_m=1_epslimits=-1.0~0.0_q0limits=0.0~3.141592653589793", J=1)
#calculate_1d_q("[sz][minimal]_lamb=0.01_r=(0, 1, 1)_Nq0=150_Neps=150_Nq=100_tx=0.5_ty=0.5_tz=0.5_m=1_epslimits=-1.0~0.0_q0limits=0.0~3.141592653589793", "[sz][minimal]_lamb=0.01_r=(0, -1, -1)_Nq0=150_Neps=150_Nq=100_tx=0.5_ty=0.5_tz=0.5_m=1_epslimits=-1.0~0.0_q0limits=0.0~3.141592653589793", J=1)

#calculate_1d_q("[s0][minimal]lamb=0.01_r=(1, 0, 1)_Nq0=150_Neps=750_Nq=100_tx=0.5_tx=0.5_tx=0.5_epslimits=-5~0_q0limits=0.0~3.141592653589793", "[s0][minimal]lamb=0.01_r=(-1, 0, -1)_Nq0=150_Neps=750_Nq=100_tx=0.5_tx=0.5_tx=0.5_epslimits=-5~0_q0limits=0.0~3.141592653589793", J=1)
#calculate_1d_q("[sy][minimal]lamb=0.01_r=(1, 1, 0)_Nq0=150_Neps=750_Nq=100_tx=0.5_tx=0.5_tx=0.5_epslimits=-5~0_q0limits=0.0~3.141592653589793", "[sy][minimal]lamb=0.01_r=(-1, -1, 0)_Nq0=150_Neps=750_Nq=100_tx=0.5_tx=0.5_tx=0.5_epslimits=-5~0_q0limits=0.0~3.141592653589793", J=1)
#calculate_1d_q("[sz][minimal]lamb=0.01_r=(0, 1, 1)_Nq0=150_Neps=750_Nq=100_tx=0.5_tx=0.5_tx=0.5_epslimits=-5~0_q0limits=0.0~3.141592653589793", "[sz][minimal]lamb=0.01_r=(0, -1, -1)_Nq0=150_Neps=750_Nq=100_tx=0.5_tx=0.5_tx=0.5_epslimits=-5~0_q0limits=0.0~3.141592653589793", J=1)
