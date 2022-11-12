# Copyright 2022 Hyun-Yong Lee

import numpy as np
from tenpy.models.lattice import Site, Chain, Square
from tenpy.models.model import CouplingModel, NearestNeighborModel, MPOModel, CouplingMPOModel
from tenpy.models import lattice
from tenpy.tools.params import Config
from tenpy.networks.site import SpinHalfSite  # if you want to use the predefined site
import matplotlib.pyplot as plt
__all__ = ['KITAEV_ELECTRIC_FIELD']


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

        basis = [ [np.sqrt(3.),0], [0,3] ]
        pos = [ [0,0], [-np.sqrt(3.)*0.5,0.5], [-np.sqrt(3.)*0.5,1.5], [0,2] ]
        nn = [ (0, 1, [0,0]), (0, 1, [1,0]), (1, 2, [0,0]), (3, 2, [1,0]), (2, 3, [0,0]), (3, 0, [0,1]) ] 
        lat = lattice.Lattice( Ls=[Lx, Ly], 
                              unit_cell=[site, site, site, site], 
                              basis=basis,
                              positions=pos, 
                              bc=bc, 
                              bc_MPS=bc_MPS)
        lat.pairs['nearest_neighbors'] = nn

        CouplingModel.__init__(self, lat)

        v_on = np.asarray(np.ones(lat.coupling_shape([0,0])[0]))
        v_on[Lx-1,:] = 0.

        v_ver = np.asarray(np.ones(lat.coupling_shape([0,1])[0]))
        v_ver[Lx-1,:] = 0.


        a = np.asarray(np.zeros([Lx,Ly]))
        a[Lx-1,:] = 1.
        self.add_onsite( -1.*a, 0, 'Sigmax')
        self.add_onsite( -1.*a, 3, 'Sigmax')
        
        # Kitaev interaction
        # x-bond
        self.add_coupling( -K*v_on, 3, 'Sigmax', 2, 'Sigmax', [0,0])
        self.add_coupling( -K, 1, 'Sigmax', 0, 'Sigmax', [-1,0])
        
        # y-bond
        self.add_coupling( -K*v_on, 1, 'Sigmay', 0, 'Sigmay', [0,0])
        self.add_coupling( -K, 3, 'Sigmay', 2, 'Sigmay', [1,0])
        
        # z-bond
        self.add_coupling( -K, 1, 'Sigmaz', 2, 'Sigmaz', [0,0])
        self.add_coupling( -K*v_ver, 3, 'Sigmaz', 0, 'Sigmaz', [0,1])

        
        # Electric field (b-direction)  A->B: positive
        # x-bond
        self.add_coupling( (-1.)*(-Eb)/np.sqrt(2.)*v_on, 3, 'Sigmay', 2, 'Sigmaz', [0, 0])
        self.add_coupling( (+1.)*(-Eb)/np.sqrt(2.)*v_on, 3, 'Sigmaz', 2, 'Sigmay', [0, 0])
        self.add_coupling( (+1.)*(-Eb)/np.sqrt(2.)*v_on, 3, 'Sigmaz', 2, 'Sigmax', [0, 0])
        self.add_coupling( (-1.)*(-Eb)/np.sqrt(2.)*v_on, 3, 'Sigmax', 2, 'Sigmaz', [0, 0])

        self.add_coupling( (-1.)*(-Eb)/np.sqrt(2.), 1, 'Sigmay', 0, 'Sigmaz', [-1, 0])
        self.add_coupling( (+1.)*(-Eb)/np.sqrt(2.), 1, 'Sigmaz', 0, 'Sigmay', [-1, 0])
        self.add_coupling( (+1.)*(-Eb)/np.sqrt(2.), 1, 'Sigmaz', 0, 'Sigmax', [-1, 0])
        self.add_coupling( (-1.)*(-Eb)/np.sqrt(2.), 1, 'Sigmax', 0, 'Sigmaz', [-1, 0])
        
        # y-bond
        self.add_coupling( (-1.)*(-Eb)/np.sqrt(2.)*v_on, 1, 'Sigmay', 0, 'Sigmaz', [0, 0])
        self.add_coupling( (+1.)*(-Eb)/np.sqrt(2.)*v_on, 1, 'Sigmaz', 0, 'Sigmay', [0, 0])
        self.add_coupling( (+1.)*(-Eb)/np.sqrt(2.)*v_on, 1, 'Sigmaz', 0, 'Sigmax', [0, 0])
        self.add_coupling( (-1.)*(-Eb)/np.sqrt(2.)*v_on, 1, 'Sigmax', 0, 'Sigmaz', [0, 0])

        self.add_coupling( (-1.)*(-Eb)/np.sqrt(2.), 3, 'Sigmay', 2, 'Sigmaz', [1, 0])
        self.add_coupling( (+1.)*(-Eb)/np.sqrt(2.), 3, 'Sigmaz', 2, 'Sigmay', [1, 0])
        self.add_coupling( (+1.)*(-Eb)/np.sqrt(2.), 3, 'Sigmaz', 2, 'Sigmax', [1, 0])
        self.add_coupling( (-1.)*(-Eb)/np.sqrt(2.), 3, 'Sigmax', 2, 'Sigmaz', [1, 0])
        
        # z-bond
        self.add_coupling( (-1.)*(-Eb)/np.sqrt(2.), 1, 'Sigmay', 2, 'Sigmaz', [0, 0])
        self.add_coupling( (+1.)*(-Eb)/np.sqrt(2.), 1, 'Sigmaz', 2, 'Sigmay', [0, 0])
        self.add_coupling( (+1.)*(-Eb)/np.sqrt(2.), 1, 'Sigmaz', 2, 'Sigmax', [0, 0])
        self.add_coupling( (-1.)*(-Eb)/np.sqrt(2.), 1, 'Sigmax', 2, 'Sigmaz', [0, 0])

        self.add_coupling( (-1.)*(-Eb)/np.sqrt(2.)*v_ver, 3, 'Sigmay', 0, 'Sigmaz', [0, 1])
        self.add_coupling( (+1.)*(-Eb)/np.sqrt(2.)*v_ver, 3, 'Sigmaz', 0, 'Sigmay', [0, 1])
        self.add_coupling( (+1.)*(-Eb)/np.sqrt(2.)*v_ver, 3, 'Sigmaz', 0, 'Sigmax', [0, 1])
        self.add_coupling( (-1.)*(-Eb)/np.sqrt(2.)*v_ver, 3, 'Sigmax', 0, 'Sigmaz', [0, 1])


        # Electric field (c-direction)  A->B: positive
        # x-bond
        self.add_coupling( (+1.)*(-Ec)/np.sqrt(3.)*v_on, 3, 'Sigmax', 2, 'Sigmay', [0, 0])
        self.add_coupling( (-1.)*(-Ec)/np.sqrt(3.)*v_on, 3, 'Sigmay', 2, 'Sigmax', [0, 0])
        self.add_coupling( (+1.)*(-Ec)/np.sqrt(3.)*v_on, 3, 'Sigmay', 2, 'Sigmaz', [0, 0])
        self.add_coupling( (-1.)*(-Ec)/np.sqrt(3.)*v_on, 3, 'Sigmaz', 2, 'Sigmay', [0, 0])
        self.add_coupling( (+1.)*(-Ec)/np.sqrt(3.)*v_on, 3, 'Sigmaz', 2, 'Sigmax', [0, 0])
        self.add_coupling( (-1.)*(-Ec)/np.sqrt(3.)*v_on, 3, 'Sigmax', 2, 'Sigmaz', [0, 0])

        self.add_coupling( (+1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmax', 0, 'Sigmay', [-1, 0])
        self.add_coupling( (-1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmay', 0, 'Sigmax', [-1, 0])
        self.add_coupling( (+1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmay', 0, 'Sigmaz', [-1, 0])
        self.add_coupling( (-1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmaz', 0, 'Sigmay', [-1, 0])
        self.add_coupling( (+1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmaz', 0, 'Sigmax', [-1, 0])
        self.add_coupling( (-1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmax', 0, 'Sigmaz', [-1, 0])
        
        # y-bond
        self.add_coupling( (+1.)*(-Ec)/np.sqrt(3.)*v_on, 1, 'Sigmax', 0, 'Sigmay', [0, 0])
        self.add_coupling( (-1.)*(-Ec)/np.sqrt(3.)*v_on, 1, 'Sigmay', 0, 'Sigmax', [0, 0])
        self.add_coupling( (+1.)*(-Ec)/np.sqrt(3.)*v_on, 1, 'Sigmay', 0, 'Sigmaz', [0, 0])
        self.add_coupling( (-1.)*(-Ec)/np.sqrt(3.)*v_on, 1, 'Sigmaz', 0, 'Sigmay', [0, 0])
        self.add_coupling( (+1.)*(-Ec)/np.sqrt(3.)*v_on, 1, 'Sigmaz', 0, 'Sigmax', [0, 0])
        self.add_coupling( (-1.)*(-Ec)/np.sqrt(3.)*v_on, 1, 'Sigmax', 0, 'Sigmaz', [0, 0])
        
        self.add_coupling( (+1.)*(-Ec)/np.sqrt(3.), 3, 'Sigmax', 2, 'Sigmay', [1, 0])
        self.add_coupling( (-1.)*(-Ec)/np.sqrt(3.), 3, 'Sigmay', 2, 'Sigmax', [1, 0])
        self.add_coupling( (+1.)*(-Ec)/np.sqrt(3.), 3, 'Sigmay', 2, 'Sigmaz', [1, 0])
        self.add_coupling( (-1.)*(-Ec)/np.sqrt(3.), 3, 'Sigmaz', 2, 'Sigmay', [1, 0])
        self.add_coupling( (+1.)*(-Ec)/np.sqrt(3.), 3, 'Sigmaz', 2, 'Sigmax', [1, 0])
        self.add_coupling( (-1.)*(-Ec)/np.sqrt(3.), 3, 'Sigmax', 2, 'Sigmaz', [1, 0])
        
        # z-bond
        self.add_coupling( (+1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmax', 2, 'Sigmay', [0, 0])
        self.add_coupling( (-1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmay', 2, 'Sigmax', [0, 0])
        self.add_coupling( (+1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmay', 2, 'Sigmaz', [0, 0])
        self.add_coupling( (-1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmaz', 2, 'Sigmay', [0, 0])
        self.add_coupling( (+1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmaz', 2, 'Sigmax', [0, 0])
        self.add_coupling( (-1.)*(-Ec)/np.sqrt(3.), 1, 'Sigmax', 2, 'Sigmaz', [0, 0])
        
        self.add_coupling( (+1.)*(-Ec)/np.sqrt(3.)*v_ver, 3, 'Sigmax', 0, 'Sigmay', [0, 1])
        self.add_coupling( (-1.)*(-Ec)/np.sqrt(3.)*v_ver, 3, 'Sigmay', 0, 'Sigmax', [0, 1])
        self.add_coupling( (+1.)*(-Ec)/np.sqrt(3.)*v_ver, 3, 'Sigmay', 0, 'Sigmaz', [0, 1])
        self.add_coupling( (-1.)*(-Ec)/np.sqrt(3.)*v_ver, 3, 'Sigmaz', 0, 'Sigmay', [0, 1])
        self.add_coupling( (+1.)*(-Ec)/np.sqrt(3.)*v_ver, 3, 'Sigmaz', 0, 'Sigmax', [0, 1])
        self.add_coupling( (-1.)*(-Ec)/np.sqrt(3.)*v_ver, 3, 'Sigmax', 0, 'Sigmaz', [0, 1])

        # on-site
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite( +hb/np.sqrt(2.), u, 'Sigmax')
            self.add_onsite( -hb/np.sqrt(2.), u, 'Sigmay')
            
            self.add_onsite( -hc/np.sqrt(3.), u, 'Sigmax')
            self.add_onsite( -hc/np.sqrt(3.), u, 'Sigmay')
            self.add_onsite( -hc/np.sqrt(3.), u, 'Sigmaz')
             
        
        
        MPOModel.__init__(self, lat, self.calc_H_MPO())


