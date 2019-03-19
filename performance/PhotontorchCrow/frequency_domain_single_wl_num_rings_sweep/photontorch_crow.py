import torch
import numpy as np
from photontorch import Component, Network, Source, Detector, Waveguide, DirectionalCouplerWithLength, DirectionalCoupler
from photontorch.torch_ext.nn import Parameter, Buffer
class _DC(Component):
    """
    Terms:
        0    2
         \  /
          ||
         /  \
        1    3
    """
    num_ports = 4
    def __init__(self, coupling=0.5, trainable=True):
        super(_DC, self).__init__(name=None)
        parameter = Parameter if trainable else Buffer
        self._alpha = parameter(
            torch.tensor(data=np.arccos(coupling), dtype=torch.get_default_dtype())
        )
    @property
    def coupling(self):
        return torch.cos(self._alpha)
    def get_S(self):
        # Coupling & Transmission
        k = torch.cos(self._alpha)[None] # make 1D
        t = torch.sin(self._alpha)[None] # make 1D
        
        # S matrix
        S = torch.zeros((2, self.env.num_wavelengths, 4, 4))
        # 0 <-> 1
        S[0,:,0,1] = S[0,:,1,0] =  t
        # 0 <-> 3
        S[1,:,0,3] = S[1,:,3,0] =  k
        # 1 <-> 2
        S[1,:,1,2] = S[1,:,2,1] =  k
        # 2 <-> 3
        S[0,:,2,3] = S[0,:,3,2] =  t
        
        # return S-matrix
        return S

class PhotontorchCrow_old(Network):
    def __init__(
        self,
        num_rings=1,
        ring_length=1e-5,
        loss=0,
        neff=2.34,
        ng=3.4,
        wl0=1.55e-6,
        trainable=True,
        couplings=None,
        name=None
    ):
        self.num_rings = num_rings
        if couplings is None:
            couplings = 0.5*np.ones(num_rings+1)
        components = {}
        for i in range(num_rings+1):
            components['elem%i'%i] = DirectionalCouplerWithLength(
                dc=_DC(coupling=couplings[i], trainable=trainable),
                wg= Waveguide(
                    length=0.5*ring_length,
                    loss=loss,
                    neff=neff,
                    ng=ng,
                    wl0=wl0,
                    trainable=False,
                ),
            )
            
        connections = []
        for i in range(num_rings):
            connections += ['elem%i:2:elem%i:0'%(i, i+1)]
            connections += ['elem%i:3:elem%i:1'%(i, i+1)]
        
        super(PhotontorchCrow, self).__init__(components, connections, name=name)
        
    def terminate(self, term=None):
        if term is None:
            name1 = 'drop' if self.num_rings%2 else 'add'
            name2 = 'add' if self.num_rings%2 else 'drop'
            term = [Source('in'), Detector('pass'), Detector(name1), Detector(name2)]
        return super(PhotontorchCrow, self).terminate(term=term)
    
    
class PhotontorchCrow(Network):
    def __init__(
        self,
        num_rings=1,
        ring_length=1e-5,
        loss=0,
        neff=2.34,
        ng=3.4,
        wl0=1.55e-6,
        trainable=True,
        couplings=None,
        name=None
    ):
        self.num_rings = num_rings
        if couplings is None:
            couplings = 0.5*np.ones(num_rings+1)
        
            
        comps = {}
        comps["src"] = Source()
        comps["det1"] = Detector()
        comps["det2"] = Detector()
        comps["det3"] = Detector()
        
        conns = []
        for i in range(num_rings + 1):
            comps["dc_%i"%i] = DirectionalCoupler(coupling=couplings[i], trainable=trainable)
            for j in range(4):
                comps["wg_%i_%i"%(i, j)] = Waveguide(length=0.25*ring_length, loss=loss, neff=neff, ng=ng, wl0=wl0, trainable=False)
            conns.append("dc_%i:0:wg_%i_%i:1"%(i, i, 0))
            conns.append("dc_%i:1:wg_%i_%i:1"%(i, i, 1))
            conns.append("dc_%i:2:wg_%i_%i:0"%(i, i, 2))
            conns.append("dc_%i:3:wg_%i_%i:0"%(i, i, 3))
        
        for i in range(num_rings):
            conns.append("wg_%i_%i:1:wg_%i_%i:0"%(i, 2, i+1, 0))
            conns.append("wg_%i_%i:1:wg_%i_%i:0"%(i, 3, i+1, 1))
        
        conns.append("wg_0_0:0:src:0")
        conns.append("wg_0_1:0:det1:0")
        conns.append("wg_%i_2:1:det2:0"%num_rings)
        conns.append("wg_%i_3:1:det3:0"%num_rings)
            
        super(PhotontorchCrow, self).__init__(comps, conns, name=name)
        
    def terminate(self, term=None):
        return self
