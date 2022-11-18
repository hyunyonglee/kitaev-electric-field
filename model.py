# Copyright 2020 Hyun-Yong Lee

import numpy as np
from tenpy.models.lattice import Site, Ladder
from tenpy.models.lattice import Honeycomb
from tenpy.models.model import CouplingModel, NearestNeighborModel, MPOModel, CouplingMPOModel
from tenpy.linalg import np_conserved as npc
from tenpy.tools.params import Config
from tenpy.networks.site import SpinHalfSite  # if you want to use the predefined site
import matplotlib.pyplot as plt
__all__ = ['KITAEV_TFI']


class KITAEV_ELECTRIC_FIELD(CouplingModel,MPOModel):
    
    def __init__(self, model_params):
        
        # 0) read out/set default parameters 
        if not isinstance(model_params, Config):
            model_params = Config(model_params, "KITAEV_ELECTRIC_FIELD")
        Lx = model_params.get('Lx', 1)
        Ly = model_params.get('Ly', 1)
        K = model_params.get('K', 1.)
        hb = model_params.get('hb', 0.)
        hc = model_params.get('hc', 0.)
        Eb = model_params.get('Eb', 0.)
        Ec = model_params.get('Ec', 0.)
        bc_MPS = model_params.get('bc_MPS', 'finite')
        bc = model_params.get('bc',['open','periodic'])
        
        site = SpinHalfSite(conserve=None)
        lat = Honeycomb(Lx=Lx, Ly=Ly, sites=site, bc=bc, bc_MPS=bc_MPS, order='Cstyle')
        
        CouplingModel.__init__(self, lat)


        ##################
        # Magnetic field #
        ##################
        for u in range(len(self.lat.unit_cell)):

            self.add_onsite( +hb/np.sqrt(2.), u, 'Sigmaz')
            self.add_onsite( -hb/np.sqrt(2.), u, 'Sigmax')
            # self.add_onsite( +hb/np.sqrt(2.), u, 'Sigmax')
            # self.add_onsite( -hb/np.sqrt(2.), u, 'Sigmay')
            
            self.add_onsite( -hc/np.sqrt(3.), u, 'Sigmax')
            self.add_onsite( -hc/np.sqrt(3.), u, 'Sigmay')
            self.add_onsite( -hc/np.sqrt(3.), u, 'Sigmaz')
        

        ##########
        # Kitaev #
        ##########
        # x-bond
        self.add_coupling( -K, 1, 'Sigmax', 0, 'Sigmax', [0,0])
        
        # y-bond
        self.add_coupling( -K, 1, 'Sigmay', 0, 'Sigmay', [1,0])
        
        # z-bond
        self.add_coupling( -K, 1, 'Sigmaz', 0, 'Sigmaz', [0,1])
        

        ################################################
        # Electric field (b-direction)  A->B: positive #
        ################################################
        # x-bond
        self.add_coupling( (-1.)*(-Eb)/np.sqrt(2.), 1, 'Sigmay', 0, 'Sigmaz', [0,0])
        self.add_coupling( (+1.)*(-Eb)/np.sqrt(2.), 1, 'Sigmaz', 0, 'Sigmay', [0,0])
        self.add_coupling( (+1.)*(-Eb)/np.sqrt(2.), 1, 'Sigmaz', 0, 'Sigmax', [0,0])
        self.add_coupling( (-1.)*(-Eb)/np.sqrt(2.), 1, 'Sigmax', 0, 'Sigmaz', [0,0])

        # y-bond
        self.add_coupling( (-1.)*(-Eb)/np.sqrt(2.), 1, 'Sigmay', 0, 'Sigmaz', [1,0])
        self.add_coupling( (+1.)*(-Eb)/np.sqrt(2.), 1, 'Sigmaz', 0, 'Sigmay', [1,0])
        self.add_coupling( (+1.)*(-Eb)/np.sqrt(2.), 1, 'Sigmaz', 0, 'Sigmax', [1,0])
        self.add_coupling( (-1.)*(-Eb)/np.sqrt(2.), 1, 'Sigmax', 0, 'Sigmaz', [1,0])
        
        # z-bond
        self.add_coupling( (-1.)*(-Eb)/np.sqrt(2.), 1, 'Sigmay', 0, 'Sigmaz', [0,1])
        self.add_coupling( (+1.)*(-Eb)/np.sqrt(2.), 1, 'Sigmaz', 0, 'Sigmay', [0,1])
        self.add_coupling( (+1.)*(-Eb)/np.sqrt(2.), 1, 'Sigmaz', 0, 'Sigmax', [0,1])
        self.add_coupling( (-1.)*(-Eb)/np.sqrt(2.), 1, 'Sigmax', 0, 'Sigmaz', [0,1])


        ################################################
        # Electric field (c-direction)  A->B: positive #
        ################################################
        # x-bond
        self.add_coupling( (+1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmax', 0, 'Sigmay', [0,0])
        self.add_coupling( (-1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmay', 0, 'Sigmax', [0,0])
        self.add_coupling( (+1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmay', 0, 'Sigmaz', [0,0])
        self.add_coupling( (-1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmaz', 0, 'Sigmay', [0,0])
        self.add_coupling( (+1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmaz', 0, 'Sigmax', [0,0])
        self.add_coupling( (-1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmax', 0, 'Sigmaz', [0,0])

        # y-bond
        self.add_coupling( (+1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmax', 0, 'Sigmay', [1,0])
        self.add_coupling( (-1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmay', 0, 'Sigmax', [1,0])
        self.add_coupling( (+1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmay', 0, 'Sigmaz', [1,0])
        self.add_coupling( (-1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmaz', 0, 'Sigmay', [1,0])
        self.add_coupling( (+1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmaz', 0, 'Sigmax', [1,0])
        self.add_coupling( (-1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmax', 0, 'Sigmaz', [1,0])

        # z-,1bond
        self.add_coupling( (+1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmax', 0, 'Sigmay', [0,1])
        self.add_coupling( (-1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmay', 0, 'Sigmax', [0,1])
        self.add_coupling( (+1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmay', 0, 'Sigmaz', [0,1])
        self.add_coupling( (-1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmaz', 0, 'Sigmay', [0,1])
        self.add_coupling( (+1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmaz', 0, 'Sigmax', [0,1])
        self.add_coupling( (-1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmax', 0, 'Sigmaz', [0,1])

        
        MPOModel.__init__(self, lat, self.calc_H_MPO())



class KITAEV_ELECTRIC_FIELD_RHOMBIC2(CouplingModel,MPOModel):
    
    def __init__(self, model_params):
        
        # 0) read out/set default parameters 
        if not isinstance(model_params, Config):
            model_params = Config(model_params, "KITAEV_ELECTRIC_FIELD_RHOMBIC2")
        Lx = model_params.get('Lx', 1)
        Ly = model_params.get('Ly', 1)
        K = model_params.get('K', 1.)
        hb = model_params.get('hb', 0.)
        hc = model_params.get('hc', 0.)
        Eb = model_params.get('Eb', 0.)
        Ec = model_params.get('Ec', 0.)
        bc_MPS = model_params.get('bc_MPS', 'infinite')
        bc = model_params.get('bc','periodic')
        
        site = SpinHalfSite(conserve=None)
        lat = Ladder(L=1, 
                    sites=[site, site], 
                    bc='periodic', 
                    bc_MPS='infinite')
        
        CouplingModel.__init__(self, lat)

        ##################
        # Magnetic field #
        ##################
        for u in range(len(self.lat.unit_cell)):

            self.add_onsite( +hb/np.sqrt(2.), u, 'Sigmaz')
            self.add_onsite( -hb/np.sqrt(2.), u, 'Sigmax')
            # self.add_onsite( +hb/np.sqrt(2.), u, 'Sigmax')
            # self.add_onsite( -hb/np.sqrt(2.), u, 'Sigmay')
            
            self.add_onsite( -hc/np.sqrt(3.), u, 'Sigmax')
            self.add_onsite( -hc/np.sqrt(3.), u, 'Sigmay')
            self.add_onsite( -hc/np.sqrt(3.), u, 'Sigmaz')

   
        ##########
        # Kitaev #
        ##########
        # x-bond
        self.add_coupling( -K, 1, 'Sigmax', 0, 'Sigmax', 0)
        
        # y-bond
        self.add_coupling( -K, 1, 'Sigmay', 0, 'Sigmay', 5)
        
        # z-bond
        self.add_coupling( -K, 1, 'Sigmaz', 0, 'Sigmaz', 1)
        

        ################################################
        # Electric field (b-direction)  A->B: positive #
        ################################################
        # x-bond
        self.add_coupling( (-1.)*(-Eb)/np.sqrt(2.), 1, 'Sigmay', 0, 'Sigmaz', 0)
        self.add_coupling( (+1.)*(-Eb)/np.sqrt(2.), 1, 'Sigmaz', 0, 'Sigmay', 0)
        self.add_coupling( (+1.)*(-Eb)/np.sqrt(2.), 1, 'Sigmaz', 0, 'Sigmax', 0)
        self.add_coupling( (-1.)*(-Eb)/np.sqrt(2.), 1, 'Sigmax', 0, 'Sigmaz', 0)

        # y-bond
        self.add_coupling( (-1.)*(-Eb)/np.sqrt(2.), 1, 'Sigmay', 0, 'Sigmaz', 5)
        self.add_coupling( (+1.)*(-Eb)/np.sqrt(2.), 1, 'Sigmaz', 0, 'Sigmay', 5)
        self.add_coupling( (+1.)*(-Eb)/np.sqrt(2.), 1, 'Sigmaz', 0, 'Sigmax', 5)
        self.add_coupling( (-1.)*(-Eb)/np.sqrt(2.), 1, 'Sigmax', 0, 'Sigmaz', 5)
        
        # z-bond
        self.add_coupling( (-1.)*(-Eb)/np.sqrt(2.), 1, 'Sigmay', 0, 'Sigmaz', 1)
        self.add_coupling( (+1.)*(-Eb)/np.sqrt(2.), 1, 'Sigmaz', 0, 'Sigmay', 1)
        self.add_coupling( (+1.)*(-Eb)/np.sqrt(2.), 1, 'Sigmaz', 0, 'Sigmax', 1)
        self.add_coupling( (-1.)*(-Eb)/np.sqrt(2.), 1, 'Sigmax', 0, 'Sigmaz', 1)


        ################################################
        # Electric field (c-direction)  A->B: positive #
        ################################################
        # x-bond
        self.add_coupling( (+1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmax', 0, 'Sigmay', 0)
        self.add_coupling( (-1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmay', 0, 'Sigmax', 0)
        self.add_coupling( (+1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmay', 0, 'Sigmaz', 0)
        self.add_coupling( (-1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmaz', 0, 'Sigmay', 0)
        self.add_coupling( (+1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmaz', 0, 'Sigmax', 0)
        self.add_coupling( (-1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmax', 0, 'Sigmaz', 0)

        # y-bond
        self.add_coupling( (+1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmax', 0, 'Sigmay', 5)
        self.add_coupling( (-1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmay', 0, 'Sigmax', 5)
        self.add_coupling( (+1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmay', 0, 'Sigmaz', 5)
        self.add_coupling( (-1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmaz', 0, 'Sigmay', 5)
        self.add_coupling( (+1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmaz', 0, 'Sigmax', 5)
        self.add_coupling( (-1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmax', 0, 'Sigmaz', 5)

        # z-,1bond
        self.add_coupling( (+1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmax', 0, 'Sigmay', 1)
        self.add_coupling( (-1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmay', 0, 'Sigmax', 1)
        self.add_coupling( (+1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmay', 0, 'Sigmaz', 1)
        self.add_coupling( (-1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmaz', 0, 'Sigmay', 1)
        self.add_coupling( (+1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmaz', 0, 'Sigmax', 1)
        self.add_coupling( (-1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmax', 0, 'Sigmaz', 1)

        MPOModel.__init__(self, lat, self.calc_H_MPO())
        
        

        