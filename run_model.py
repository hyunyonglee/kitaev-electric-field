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
import pickle



def ensure_dir(f):
    d=os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
    return d


def measurements(psi):
    
    ensure_dir("observables/")
    ensure_dir("entanglements/")
    ensure_dir("mps/")

    # Measurements
    Mx = psi.expectation_value("Sigmax")
    My = psi.expectation_value("Sigmay")
    Mz = psi.expectation_value("Sigmaz")
    EE = psi.entanglement_entropy()
    
    # Measurements - Flux
    Fs = []
    for i in range(Lx-1):
        I0 = 4*Ly*i
        for j in range(Ly):
        
            flux = psi.expectation_value_term([('Sigmaz',I0+4*j),('Sigmax',I0+4*j+1),('Sigmay',I0+4*j+2),('Sigmaz',I0+4*j+3),('Sigmax',I0+4*(j+Ly)+2),('Sigmay',I0+4*(j+Ly)+1)])
            Fs.append(flux)       
            if i<Lx-2:
                r = 0 if j < Ly-1 else 4*Ly
                flux = psi.expectation_value_term([('Sigmax',I0+4*j+3),('Sigmay',I0+4*j+4-r),('Sigmaz',I0+4*(j+Ly)+5-r),('Sigmax',I0+4*(j+Ly)+4-r),('Sigmay',I0+4*(j+Ly)+3),('Sigmaz',I0+4*(j+Ly)+2)])
                Fs.append(flux)


    return Mx, My, Mz, EE, Fs

    
def writing_file(psi, state, Mx, My, Mz, EE, Fs, K, hb, hc, Eb, Ec):

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
    
    with open('mps/%s_K_%.2f_hb_%.2f_hc%.2f_Eb%.2f_Ec%.2f.pkl' % (state,K,hb,hc,Eb,Ec), 'wb') as f:
        pickle.dump(psi, f)




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
    parser.add_argument("--Lx", help="Length of Cylinder")
    parser.add_argument("--Ly", help="Circumference of Cylinder")
    parser.add_argument("--K", help="Kitaev interaction")
    parser.add_argument("--hb", help="Magnetic field along b-direction")
    parser.add_argument("--hc", help="Magnetic field along c-direction")
    parser.add_argument("--Eb", help="Electric field along b-direction")
    parser.add_argument("--Ec", help="Electric field along c-direction")
    parser.add_argument("--CHI", help="Bond dimension")
    parser.add_argument("--RM", help="'On': randomize initial state")
    parser.add_argument("--TOL", help="Convergence criteria for Entanglent Entropy")
    parser.add_argument("--EXC", help="'On': calculate the 1st excited state")
    args=parser.parse_args()

    #python run_model.py --Lx=1 --Ly=2 --CHI=10 --hb=0.01 --K=1.0 --RM=None --TOL=1.0e-6 --EXC=None

    Lx = int(args.Lx)
    Ly = int(args.Ly)
    K = float(args.K)
    hb = float(args.hb)
    hc = float(args.hc)
    Eb = float(args.Eb)
    Ec = float(args.Ec)
    CHI = int(args.CHI)
    RM = args.RM
    TOL = float(args.TOL)
    EXC = args.EXC

    model_params = {
        "Lx": Lx,
        "Ly": Ly,
        "K": K,
        "hb": hb,
        "hc": hc,
        "Eb": Eb,
        "Ec": Ec
    }

    print("\n\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    dmrg_params = {
        # 'mixer': True,  # setting this to True helps to escape local minima
        'mixer' : dmrg.SubspaceExpansion,
        'mixer_params': {
            'amplitude': 1.e-3,
            'decay': 2.0,
            'disable_after': 30
        },
        'trunc_params': {
            'chi_max': CHI,
            'svd_min': 1.e-9
        },
        # 'lanczos_params': {
        #         'N_min': 5,
        #         'N_max': 20
        # },
        'chi_list': {0: 32, 5: 64, 10: CHI},
        'max_E_err': 1.0e-8,
        'max_S_err': TOL,
        'max_sweeps': 500,
        'combine' : True
    }

    # defining model
    M = model.KITAEV_ELECTRIC_FIELD(model_params)
    
    # defining initial state
    product_state = ["up"] * M.lat.N_sites
    psi0 = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
    
    # randomization of initial state
    if RM == 'On':
        TEBD_params = {'N_steps': 4, 'trunc_params':{'chi_max': 4}, 'verbose': 0}
        eng = tebd.RandomUnitaryEvolution(psi0, TEBD_params)
        eng.run()
        psi0.canonical_form() 
    psi1 = psi0.copy()

    # ground state
    eng = dmrg.TwoSiteDMRGEngine(psi0, M, dmrg_params)
    E0, psi0 = eng.run()  # equivalent to dmrg.run() up to the return parameters.
    psi0.canonical_form() 
    Mx0, My0, Mz0, EE0, Fs0 = measurements(psi0)
    writing_file(psi0, "gs", Mx0, My0, Mz0, EE0, Fs0, K, hb, hc, Eb, Ec)


    # excited state
    if EXC == 'On':
        dmrg_params['orthogonal_to'] = [psi0]
        eng1 = dmrg.TwoSiteDMRGEngine(psi1, M, dmrg_params)
        E1, psi1 = eng1.run()  # equivalent to dmrg.run() up to the return parameters.
        psi1.canonical_form() 
        Mx1, My1, Mz1, EE1, Fs1 = measurements(psi1)
        writing_file(psi1, "exc", Mx1, My1, Mz1, EE1, Fs1, K, hb, hc, Eb, Ec)
        gap = E1 - E0

    else:
        gap = 0.
    #

    file_observables = open("observables/gs_observables.txt","a")
    file_observables.write(repr(K) + " " + repr(hb) + " " + repr(hc) + " " + repr(Eb) + " " + repr(Ec) + " " + repr(E0) + " " + repr(np.mean(Mx0)) + " " + repr(np.mean(My0)) + " " + repr(np.mean(Mz0)) + " " + repr(np.mean(EE0[int(4*Ly*Lx/2)-1])) + " " + repr(np.mean(Fs0)) + " " + repr(gap) + " " + "\n")
    
    file_observables = open("observables/exc_observables.txt","a")
    file_observables.write(repr(K) + " " + repr(hb) + " " + repr(hc) + " " + repr(Eb) + " " + repr(Ec) + " " + repr(E1) + " " + repr(np.mean(Mx1)) + " " + repr(np.mean(My1)) + " " + repr(np.mean(Mz1)) + " " + repr(np.mean(EE1[int(4*Ly*Lx/2)-1])) + " " + repr(np.mean(Fs1)) + " " + repr(gap) + " " + "\n")
    
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n\n")







