# Copyright 2022 Hyun-Yong Lee

import numpy as np
import model
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
from tenpy.algorithms import tebd
from tenpy.tools.process import mkl_set_nthreads
import os
import os.path
import argparse, sys
import h5py
from tenpy.tools import hdf5_io

def ensure_dir(f):
    d=os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
    return d


def measurements(psi, bc_MPS, twist):
    
    ensure_dir("observables/")
    ensure_dir("entanglements/")
    ensure_dir("mps/")

    # Measurements
    Mx = psi.expectation_value("Sigmax")
    My = psi.expectation_value("Sigmay")
    Mz = psi.expectation_value("Sigmaz")
    EE = psi.entanglement_entropy()
    ES = psi.entanglement_spectrum()
    
    # Measurements - Flux
    Fs = []
    R = Lx if bc_MPS == 'infinite' else Lx-1
    for i in range(R):

        I0 = 2*Ly*i
        for j in range(Ly):
            
            if twist =='Off':
                r = 0 if j < Ly-1 else 2*Ly
            else:
                r = 0
            # print(I0+2*j+1,I0+2*j+2-r,I0+2*j+3-r,I0+2*Ly+2*j+2-r,I0+2*Ly+2*j+1,I0+2*Ly+2*j)
            flux = psi.expectation_value_term([('Sigmax',I0+2*j+1),('Sigmay',I0+2*j+2-r),('Sigmaz',I0+2*j+3-r),('Sigmax',I0+2*Ly+2*j+2-r),('Sigmay',I0+2*Ly+2*j+1),('Sigmaz',I0+2*Ly+2*j)])
            Fs.append(flux)       

    if twist =='Off':
        operators = ['Sigmay']*2*Ly
        i0 = 0
    else:
        operators = ['Sigmax'] + ['Sigmay']*2*(Ly-1) + ['Sigmax']
        i0 = 1
    Wl = psi.expectation_value_multi_sites(operators, i0=i0)

    return Mx, My, Mz, EE, ES, Fs, Wl

    
def writing_file(psi, state, Mx, My, Mz, EE, ES, Fs, K, hb, hc, Eb, Ec):

    # Writing
    file_W = open("observables/%s_Flux.txt" % state,"a")
    file_W.write(repr(K) + " " + repr(hb) + " " + repr(hc) + " " + repr(Eb) + " " + repr(Ec) + " " + "  ".join(map(str, Fs)) + " " + "\n")
    
    file_Mx = open("observables/%s_Mx.txt" % state,"a")
    file_Mx.write(repr(K) + " " + repr(hb) + " " + repr(hc) + " " + repr(Eb) + " " + repr(Ec) + " " + "  ".join(map(str, Mx)) + " " + "\n")
    
    file_My = open("observables/%s_My.txt" % state,"a")
    file_My.write(repr(K) + " " + repr(hb) + " " + repr(hc) + " " + repr(Eb) + " " + repr(Ec) + " " + "  ".join(map(str, My)) + " " + "\n")
    
    file_Mz = open("observables/%s_Mz.txt" % state,"a")
    file_Mz.write(repr(K) + " " + repr(hb) + " " + repr(hc) + " " + repr(Eb) + " " + repr(Ec) + " " + "  ".join(map(str, Mz)) + " " + "\n")
    
    file_EE = open("entanglements/%s_EE.txt" % state,"a")
    file_EE.write(repr(K) + " " + repr(hb) + " " + repr(hc) + " " + repr(Eb) + " " + repr(Ec) + " " + "  ".join(map(str, EE)) + " " + "\n")
    
    file_ES = open( "entanglements/%s_es_K_%.1f_hb_%.3f_hc%.3f_Eb%.3f_Ec%.3f.txt" % (state,K,hb,hc,Eb,Ec),"a")
    
    for i in range(0,len(ES)):
        file_ES.write("  ".join(map(str, ES[i][0:(np.max([64,len(ES[i])]))])) + " " + "\n")

    # with gzip.open('mps/%s_K_%.1f_hb_%.3f_hc%.3f_Eb%.3f_Ec%.3f.pkl' % (state,K,hb,hc,Eb,Ec), 'wb') as f:
    #     pickle.dump(psi, f)
    data = {"psi": psi}
    with h5py.File('mps/%s_K_%.1f_hb_%.3f_hc%.3f_Eb%.3f_Ec%.3f.h5' % (state,K,hb,hc,Eb,Ec), 'w') as f:
        hdf5_io.save_to_hdf5(f, data)



# main
if __name__=='__main__':
    mkl_set_nthreads(64)

    import logging.config
    conf = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {'custom': {'format': '%(levelname)-8s: %(message)s'}},
        'handlers': {'to_file': {'class': 'logging.FileHandler',
                                 'filename': 'log',
                                 'formatter': 'custom',
                                 'level': 'INFO',
                                 'mode': 'a'},
                    'to_stdout': {'class': 'logging.StreamHandler',
                                  'formatter': 'custom',
                                  'level': 'INFO',
                                  'stream': 'ext://sys.stdout'}},
        'root': {'handlers': ['to_stdout', 'to_file'], 'level': 'DEBUG'},
    }
    logging.config.dictConfig(conf)

    parser=argparse.ArgumentParser()
    parser.add_argument("--Lx", default='2', help="Length of Cylinder")
    parser.add_argument("--Ly", default='4', help="Circumference of Cylinder")
    parser.add_argument("--K", default='1.0', help="Kitaev interaction")
    parser.add_argument("--hb", default='0.0', help="Magnetic field along b-direction")
    parser.add_argument("--hc", default='0.0', help="Magnetic field along c-direction")
    parser.add_argument("--Eb", default='0.0', help="Electric field along b-direction")
    parser.add_argument("--Ec", default='0.0', help="Electric field along c-direction")
    parser.add_argument("--chi", default='64', help="Bond dimension")
    parser.add_argument("--rm", default='Off', help="'On': randomize initial state")
    parser.add_argument("--tol", default='1.0e-6', help="Convergence criteria for Entanglent Entropy")
    parser.add_argument("--exc", default='0ff', help="'On': calculate the 1st excited state")
    parser.add_argument("--bc_MPS", default='finite', help="'finite' or 'infinite' DMRG")
    parser.add_argument("--twist", default='Off', help="'On': twisted boundary condition along y-direction ")
    parser.add_argument("--init_state", default=None, help="Load initial state")
    args=parser.parse_args()

    Lx = int(args.Lx)
    Ly = int(args.Ly)
    K = float(args.K)
    hb = float(args.hb)
    hc = float(args.hc)
    Eb = float(args.Eb)
    Ec = float(args.Ec)
    chi = int(args.chi)
    rm = args.rm
    tol = float(args.tol)
    exc = args.exc
    bc_MPS = args.bc_MPS
    twist = args.twist
    init_state = args.init_state

    if bc_MPS == 'infinite' and twist == 'Off':
        bc = 'periodic'
        x = 2*Ly*Lx-1
    elif bc_MPS == 'infinite' and twist == 'On':
        bc = ['periodic',-1]
        # x = 2*Ly*Lx-1
        x = 0
    else:
        bc = ['open','periodic']
        x = int(2*Ly*Lx/2)-1

    model_params = {
        "Lx": Lx,
        "Ly": Ly,
        "K": K,
        "hb": hb,
        "hc": hc,
        "Eb": Eb,
        "Ec": Ec,
        "bc_MPS": bc_MPS,
        "bc": bc
    }

    print("\n\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")


    # defining model
    if twist == 'On':
        M = model.KITAEV_ELECTRIC_FIELD_RHOMBIC2(model_params)
    else:
        M = model.KITAEV_ELECTRIC_FIELD(model_params)
    
    
    # defining initial state
    if init_state:
        with h5py.File(init_state, 'r') as f:
            psi0 = hdf5_io.load_from_hdf5(f, "/psi")
        psi0.canonical_form()
        chi_list = None
    else:
        # product_state = ["up","down"] * int(M.lat.N_sites/2)
        product_state = ["up"] * M.lat.N_sites
        psi0 = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
        chi_list = {0: 32, 5: 64, 10: chi}
    
    # randomization of initial state
    if rm == 'On':
        TEBD_params = {'N_steps': 4, 'trunc_params':{'chi_max': 4}, 'verbose': 0}
        eng = tebd.RandomUnitaryEvolution(psi0, TEBD_params)
        eng.run()
        psi0.canonical_form() 
    psi1 = psi0.copy()

    # DMRG params
    dmrg_params = {
        # 'mixer': True,  # setting this to True helps to escape local minima
        'mixer' : dmrg.SubspaceExpansion,
        'mixer_params': {
            'amplitude': 1.e-2,
            'decay': 2.0,
            'disable_after': 30
        },
        'trunc_params': {
            'chi_max': chi,
            'svd_min': 1.e-9
        },
        # 'lanczos_params': {
        #         'N_min': 5,
        #         'N_max': 20
        # },
        'chi_list': chi_list,
        'max_E_err': 1.0e-8,
        'max_S_err': tol,
        'max_sweeps': 500,
        'combine' : True
    }

    # ground state
    eng = dmrg.TwoSiteDMRGEngine(psi0, M, dmrg_params)
    E0, psi0 = eng.run()  # equivalent to dmrg.run() up to the return parameters.
    Mx0, My0, Mz0, EE0, ES0, Fs0, Wl0 = measurements(psi0, bc_MPS, twist)
    writing_file(psi0, "gs", Mx0, My0, Mz0, EE0, ES0, Fs0, K, hb, hc, Eb, Ec)

    # excited state
    if bc_MPS == 'finite' and exc == 'On':
        dmrg_params['orthogonal_to'] = [psi0]
        eng1 = dmrg.TwoSiteDMRGEngine(psi1, M, dmrg_params)
        E1, psi1 = eng1.run()  # equivalent to dmrg.run() up to the return parameters.
        Mx1, My1, Mz1, EE1, ES1, Fs1, Wl1 = measurements(psi1, bc_MPS, twist)
        writing_file(psi1, "exc", Mx1, My1, Mz1, EE1, ES1, Fs1, K, hb, hc, Eb, Ec)
        gap = E1 - E0

        file_observables = open("observables/exc_observables.txt","a")
        file_observables.write(repr(K) + " " + repr(hb) + " " + repr(hc) + " " + repr(Eb) + " " + repr(Ec) + " " + repr(E1) + " " + repr(np.mean(Mx1)) + " " + repr(np.mean(My1)) + " " + repr(np.mean(Mz1)) + " " + repr(np.mean(EE1[x])) + " " + repr(np.mean(Fs1)) + " " + repr(np.mean(Wl1)) + " " + repr(gap) + " " + "\n")

    else:
        gap = 0.

    #
    if bc_MPS == 'infinite':
        xi = psi0.correlation_length()
    else:
        xi = 0.
    
    file_observables = open("observables/gs_observables.txt","a")
    file_observables.write(repr(K) + " " + repr(hb) + " " + repr(hc) + " " + repr(Eb) + " " + repr(Ec) + " " + repr(E0) + " " + repr(np.mean(Mx0)) + " " + repr(np.mean(My0)) + " " + repr(np.mean(Mz0)) + " " + repr(np.mean(EE0[x])) + " " + repr(np.mean(Fs0)) + " " + repr(np.mean(Wl0)) + " " + repr(gap) + " " + repr(xi) + " " + "\n")
    
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n\n")







