import numpy as np

import openmdao.api as om

from openaerostruct.utils.constants import grav_constant

class EnduranceForEletricPropDriven(om.ExplicitComponent):
    '''
    Computes the endurance for an aircraft with constant mass, e.g battery
    or fuel cell powered vehicles,
    driven by propellers using the computed CL, CD, weight and provided
    speed of sound, Mach number, propeller and battery to motor
    output shaft efficiencies, and total energy content available

    The equation is derived on Dr D. Raymer's book ISBN 0-930403-51-7

    Parameters
    ----------
    CL : float
        Total coefficient of lift (CL) for the lifting surface.
    CD : float
        Total coefficient of drag (CD) for the lifting surface.
    W0 : float
        The operating empty weight of the aircraft, without fuel or structural
        mass. Supplied in kg despite being a 'weight' due to convention.
    speed_of_sound : float
        The Mach speed, speed of sound, at the specified flight condition.
     Mach_number : float
        The Mach number of the aircraft at the specified flight condition.
    _structural_mass : float
        Weight of a single lifting surface's structural spar.
    eta_b2s : float
        Total system efficiency from battery to motor output shaft
    eta_p : float
        propeller efficiency
    total_energy_content : float
        Total energy available. E.g, total energy stored inside the aircraft
        battery or in the chemical bounds of hydrogen.

    Returns
    -------
    endurance: float
        Computed endurance in hours based on eq. derived by Dr. Raymer


    '''

    def initialize(self):
        self.options.declare('surfaces', types=list)

    def setup(self):
        # Inputs
        for surface in self.options['surfaces']:
            name = surface['name']
            self.add_input(name + '_structural_mass', val=1., units='kg')

        self.add_input('CL', val=0.7)
        self.add_input('CD', val=0.02)
        self.add_input('W0', val=200., units='kg')
        self.add_input('speed_of_sound', val=100., units='m/s')
        self.add_input('Mach_number', val=1.2)
        self.add_input('eta_b2s', val=0.9)
        self.add_input('eta_p', val=0.8)
        self.add_input('total_energy_content', val=3.3, units='W*h')

        # Outputs
        self.add_output('endurance', val=1., units='h')

    def setup_partials(self):
        # All partial derivatives are nonzero
        self.declare_partials('*', '*')
    
    def compute(self, inputs, outputs):
        g = grav_constant
        W0 = inputs['W0']
        a = inputs['speed_of_sound']
        M = inputs['Mach_number']
        eta_b2s = inputs['eta_b2s']
        eta_p = inputs['eta_p']
        energy = inputs['total_energy_content']

        # Loop through the surfaces and add up the structural weights
        # to get the total structural weight.
        Ws = 0.
        for surface in self.options['surfaces']:
            name = surface['name']
            Ws += inputs[name + '_structural_mass']

        CL = inputs['CL']
        CD = inputs['CD']

        outputs['endurance'] = -(CL/CD) * (energy * eta_b2s *eta_p)/((W0+Ws)*g \
            * (M*a))

    def compute_partials(self, inputs, partials):
        g = grav_constant
        W0 = inputs['W0']
        a = inputs['speed_of_sound']
        M = inputs['Mach_number']
        eta_b2s = inputs['eta_b2s']
        eta_p = inputs['eta_p']
        energy = inputs['total_energy_content']

        Ws = 0.
        for surface in self.options['surfaces']:
            name = surface['name']
            Ws += inputs[name + '_structural_mass']

        CL = inputs['CL']
        CD = inputs['CD']

        dend_dCL = -(1/CD) * energy * eta_p * eta_b2s / (W0 + Ws) / g / (M * a)
        dend_denergy = -(CL/CD) * eta_p * eta_b2s / (W0 + Ws) / g / (M * a)
        dend_deta_b2s = -(CL/CD) * energy * eta_p / (W0 + Ws) / g / (M * a)
        dend_deta_p = -(CL/CD) * energy * eta_b2s / (W0 + Ws) / g / (M * a)
        dend_dCD = CL * energy * eta_b2s * eta_p / (W0 + Ws) / g / (M * a) \
            / CD**2
        dend_dM = (CL/CD) * energy * eta_b2s * eta_p / (W0 + Ws) / g / a \
            / M**2
        dend_da = (CL/CD) * energy * eta_b2s * eta_p / (W0 + Ws) / g / M \
            / a**2
        dend_dW = (CL/CD) * energy * eta_b2s * eta_p / (M * a) / g / \
            (W0 + Ws)**2

        partials['endurance', 'CL'] = dend_dCL
        partials['endurance', 'CD'] = dend_dCD
        partials['endurance', 'W0'] = dend_dW
        partials['endurance', 'speed_of_sound'] = dend_da
        partials['endurance', 'Mach_number'] = dend_dM
        partials['endurance', 'eta_b2s'] = dend_deta_b2s
        partials['endurance', 'eta_p'] = dend_deta_p
        partials ['endurance', 'total_energy_content'] = dend_denergy
        
        for surface in self.options['surfaces']:
            name = surface['name']
            inp_name = name + '_structural_mass'
            partials['endurance', inp_name] = dend_dW



