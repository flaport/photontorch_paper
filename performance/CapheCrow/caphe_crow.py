import caphe
import numpy as np
from caphe.base import link_ports
from caphe.base import CapheNodeSolver
from caphe.base import Node as CapheNode
from caphe.base import EnvironmentObject as CapheEnvironment
from caphe.base import BaseDetector as CapheDetector
from caphelib.photonics.nodes.source import Source as _CapheSource_
from caphelib.photonics.nodes.directionalcoupler import DirectionalCoupler as CapheDirectionalCoupler

class CapheSourceModel(caphe.base.TSignalModel):
    ''' A Simple Source Model with constant amplitude 1 '''
    def calculate_S(self, node, environment, p1, p2):
        return 0
    def calculate_signals(self, node, environment, t, s_in, s_ext, y):
        s_ext[0] = 1
        
class CapheSource(_CapheSource_):
    ''' A Simple Source with constant amplitude 1 '''
    model = caphe.base.ModelProperty(default=CapheSourceModel())
    def __init__(self, *args, **kwargs):
        ## We need this init to suppress a nomodel warning.
        kwargs['suppress_warning_nomodel'] = True
        super(CapheSource, self).__init__(*args, **kwargs)
    
class CapheWaveguideModel(caphe.base.SModel):
    ''' A simple waveguide Model '''
    def calculate_S(self, node, environment, p1, p2):
        wl = environment.wavelength
        if p1 == p2:
            return 0
        phase = np.exp(1j*2*np.pi/wl*node.n_eff*node.length)
        attenuation = 10**(-node.loss_dB_m*node.length/20)
        return phase*attenuation
    
class CapheDelayedWaveguideModel(caphe.base.TSignalModel):
    ''' A delayed waveguide Model '''
    def calculate_S(self, node, environment, p1, p2):
        return 0
    def calculate_signals(self, node, environment, t, s_in, s_ext, y):
        wl = environment.wavelength
        phase = np.exp(1j*2*np.pi/wl*node.n_eff*node.length)
        attenuation = 10**(-node.loss_dB_m*node.length/20)
        transmission = attenuation*phase
        s_ext[0] = transmission*s_in[1](t-node.delay)
        s_ext[1] = transmission*s_in[0](t-node.delay)
    
class CapheWaveguide(caphe.base.Node):
    ''' A simple waveguide '''
    nr_ports = 2
    model = caphe.base.ModelProperty(default=CapheWaveguideModel())
    length = caphe.base.FloatProperty(doc="The physical length of the waveguide (using SI units).")
    n_eff = caphe.base.FloatProperty(doc="The effective index of the waveguide.")
    loss_dB_m = caphe.base.FloatProperty(doc="The loss in the waveguide, expressed in dB/m.")
    @property
    def delay(self):
        return self.length*self.n_eff/c #[s]
    
class CapheCrow(object):
    def __init__(self, num_rings=1, ring_length=1e-5, loss=0, neff=2.34, 
                 ng=3.4, wl0=1.55e-6, wg_model=None, couplings=None):
        
        if couplings is None:
            couplings = 0.5*np.ones(num_rings+1)

        dcs = np.empty(num_rings+1, dtype=object)
        wgs = np.empty((num_rings+1, 4), dtype=object)

        circuit = CapheNode(name='crow', nr_ports=0)
        s = CapheSource()
        d1 = CapheDetector()
        d2 = CapheDetector()
        d3 = CapheDetector()
        
        if wg_model is None:
            wg_model = CapheWaveguideModel()
        self._wg_model = wg_model

        for i in range(num_rings+1):
            dcs[i] = CapheDirectionalCoupler(tau=(1-couplings[i])**0.5, kappa=couplings[i]**0.5)
            wgs[i,0] = CapheWaveguide(length=0.25*ring_length, n_eff=neff, loss_dB_m=loss, model=wg_model)
            wgs[i,1] = CapheWaveguide(length=0.25*ring_length, n_eff=neff, loss_dB_m=loss, model=wg_model)
            wgs[i,2] = CapheWaveguide(length=0.25*ring_length, n_eff=neff, loss_dB_m=loss, model=wg_model)
            wgs[i,3] = CapheWaveguide(length=0.25*ring_length, n_eff=neff, loss_dB_m=loss, model=wg_model)
            link_ports(dcs[i].get_port(0), wgs[i,0].get_port(1))
            link_ports(dcs[i].get_port(2), wgs[i,1].get_port(1))
            link_ports(dcs[i].get_port(1), wgs[i,2].get_port(0))
            link_ports(dcs[i].get_port(3), wgs[i,3].get_port(0))

        for i in range(num_rings):
            link_ports(wgs[i,2].get_port(1), wgs[i+1,0].get_port(0))
            link_ports(wgs[i,3].get_port(1), wgs[i+1,1].get_port(0))

        link_ports(wgs[0,0].get_port(0), s.get_port(0))
        link_ports(wgs[0,1].get_port(0), d1.get_port(0))
        link_ports(wgs[-1,2].get_port(1), d2.get_port(0))
        link_ports(wgs[-1,3].get_port(1), d3.get_port(0))
        
        self.wgs = wgs
        self.dcs = dcs

        nodes = [s, d1, d2, d3] + list(dcs) + list(wgs.ravel())

        circuit.add_nodes(*nodes)
        
        self.circuit = circuit

        self.solver = CapheNodeSolver(circuit)
        self.solver.set_integration_method(caphe.solvers.euler)
    
    @property
    def wg_model(self):
        return self._wg_model
    @wg_model.setter
    def wg_model(self, model):
        for wg in self.wgs.ravel():
            wg.model = model
    
    def time(self, time, wavelength=1.55e-6):
        if not isinstance(self.wg_model, CapheDelayedWaveguideModel):
            self.wg_model = CapheDelayedWaveguideModel()
        wavelength = np.array(wavelength)
        if wavelength.ndim == 0:
            wavelength = wavelength[None]
        detected = []
        for wl in wavelength:
            env = CapheEnvironment(name='env', wavelength=wl)
            self.solver.set_internal_dt(time[1]-time[0])
            self.solver.solve(t0=time[0], t1=time[-1], dt=time[1]-time[0], environment=env)
            _, _, det = self.solver.get_states_and_output()
            detected.append(det)
        detected = np.stack(detected, 1)
        return abs(detected)**2
    
    def frequency(self, wavelength):
        if not isinstance(self.wg_model, CapheWaveguideModel):
            self.wg_model = CapheWaveguideModel()
        wavelength = np.array(wavelength)
        if wavelength.ndim == 0:
            wavelength = wavelength[None]
        detected = np.zeros((len(wavelength), 3), dtype=complex)
        for i, wl in enumerate(wavelength):
            environment = CapheEnvironment(name='env', wavelength=wl)
            detected[i] = self.solver.get_C_exttoin(environment=environment)[0,1:]
        return abs(detected)**2