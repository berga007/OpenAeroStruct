from __future__ import division, print_function
import numpy as np

from openaerostruct.geometry.utils import generate_mesh

from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint

from openmdao.api import IndepVarComp, Problem, Group, SqliteRecorder

# Total number of nodes to use in the spanwise (num_y) and
# chordwise (num_x) directions
num_y = 31
num_x = 3

# Create a dictionary to store options about the surface
mesh_dict = {'num_y' : num_y,
             'num_x' : num_x,
             'wing_type' : 'rect',
             'symmetry' : True,
             'span_cos_spacing' : 0.5,
             'span' : 3.11,
             'root_chord' : 0.3,
             }

mesh = generate_mesh(mesh_dict)

# Apply camber to the mesh
camber = 1 - np.linspace(-1, 1, num_x) ** 2
camber *= 0.3 * 0.05

for ind_x in range(num_x):
    mesh[ind_x, :, 2] = camber[ind_x]

zshear_cp = np.zeros(10)
zshear_cp[0] = .3

xshear_cp = np.zeros(10)
xshear_cp[0] = .15

chord_cp = np.ones(10)
chord_cp[0] = .5
chord_cp[-1] = 1.5
chord_cp[-2] = 1.3

radius_cp = 0.01  * np.ones(10)

surface = {
            # Wing definition
            'name' : 'wing',        # name of the surface
            'symmetry' : True,     # if true, model one half of wing
                                    # reflected across the plane y = 0
            'S_ref_type' : 'wetted', # how we compute the wing area,
                                     # can be 'wetted' or 'projected'
            'fem_model_type' : 'tube',

            'taper' : 0.8,
            'zshear_cp' : zshear_cp,
            'xshear_cp' : xshear_cp,
            'chord_cp' : chord_cp,
            'sweep' : 20.,
            'twist_cp' : np.array([2.5, 2.5, 5.]), #np.zeros((3)),
            'thickness_cp' : np.ones((3))*.0051,

            'radius_cp' : radius_cp,
            'mesh' : mesh,

            # Aerodynamic performance of the lifting surface at
            # an angle of attack of 0 (alpha=0).
            # These CL0 and CD0 values are added to the CL and CD
            # obtained from aerodynamic analysis of the surface to get
            # the total CL and CD.
            # These CL0 and CD0 values do not vary wrt alpha.
            'CL0' : 0.0,            # CL of the surface at alpha=0
            'CD0' : 0.015,            # CD of the surface at alpha=0

            # Airfoil properties for viscous drag calculation
            'k_lam' : 0.05,         # percentage of chord with laminar
                                    # flow, used for viscous drag
            't_over_c_cp' : np.array([0.12]),      # thickness over chord ratio
            'c_max_t' : .303,       # chordwise location of maximum (NACA0015)
                                    # thickness
            'with_viscous' : True,
            'with_wave' : False,     # if true, compute wave drag

            # Material properties taken from http://www.performance-composites.com/carbonfibre/mechanicalproperties_2.asp
            'E' : 85.e9,
            'G' : 25.e9,
            'yield' : 350.e6,
            'mrho' : 1.6e3,

            'fem_origin' : 0.35,    # normalized chordwise location of the spar
            'wing_weight_ratio' : 1.,
            'struct_weight_relief' : True,    # True to add the weight of the structure to the loads on the structure
            'distributed_fuel_weight' : False,
            # Constraints
            'exact_failure_constraint' : False, # if false, use KS function
            }

# Create the problem and assign the model group
prob = Problem()

# Add problem information as an independent variables component
indep_var_comp = IndepVarComp()
indep_var_comp.add_output('v', val=22.876, units='m/s')
indep_var_comp.add_output('alpha', val=5., units='deg')
indep_var_comp.add_output('Mach_number', val=0.071)
indep_var_comp.add_output('re', val=1.e6, units='1/m')
indep_var_comp.add_output('rho', val=0.770816, units='kg/m**3')
indep_var_comp.add_output('CT', val=9.80665 * 8.6e-6, units='1/s')
indep_var_comp.add_output('R', val=1800e3, units='m')
indep_var_comp.add_output('W0', val=10.,  units='kg')
indep_var_comp.add_output('speed_of_sound', val=322.2, units='m/s')
indep_var_comp.add_output('load_factor', val=1.)
indep_var_comp.add_output('empty_cg', val=np.array([0.2, 0., 0.]), units='m')

prob.model.add_subsystem('prob_vars',
     indep_var_comp,
     promotes=['*'])

aerostruct_group = AerostructGeometry(surface=surface)

name = 'wing'

# Add tmp_group to the problem with the name of the surface.
prob.model.add_subsystem(name, aerostruct_group,
    promotes_inputs=['load_factor'])

point_name = 'AS_point_0'

# Create the aero point group and add it to the model
AS_point = AerostructPoint(surfaces=[surface])

prob.model.add_subsystem(point_name, AS_point,
    promotes_inputs=['v', 'alpha', 'Mach_number', 're', 'rho', 'CT', 'R',
        'W0', 'speed_of_sound', 'empty_cg', 'load_factor'])

com_name = point_name + '.' + name + '_perf'
prob.model.connect(name + '.K', point_name + '.coupled.' + name + '.K')
prob.model.connect(name + '.nodes', point_name + '.coupled.' + name + '.nodes')

# Connect aerodyamic mesh to coupled group mesh
prob.model.connect(name + '.mesh', point_name + '.coupled.' + name + '.mesh')

# Connect performance calculation variables
prob.model.connect(name + '.radius', com_name + '.radius')
prob.model.connect(name + '.thickness', com_name + '.thickness')
prob.model.connect(name + '.nodes', com_name + '.nodes')
prob.model.connect(name + '.cg_location', point_name + '.' + 'total_perf.' + name + '_cg_location')
prob.model.connect(name + '.structural_weight', point_name + '.' + 'total_perf.' + name + '_structural_weight')
prob.model.connect(name + '.t_over_c', com_name + '.t_over_c')

# from openmdao.api import ScipyOptimizeDriver
# prob.driver = ScipyOptimizeDriver()
# prob.driver.options['tol'] = 1e-9

from openmdao.api import pyOptSparseDriver
prob.driver = pyOptSparseDriver()
prob.driver.options['optimizer'] = "SNOPT"
prob.driver.opt_settings['Major optimality tolerance'] = 1e-6
prob.driver.opt_settings['Major feasibility tolerance'] = 1e-8
prob.driver.opt_settings['Major iterations limit'] = 1000
prob.driver.opt_settings['Verify level'] = -1

recorder = SqliteRecorder("aerostruct.db")
prob.driver.add_recorder(recorder)
prob.driver.recording_options['record_derivatives'] = True
prob.driver.recording_options['includes'] = ['*']

# Setup problem and add design variables, constraint, and objective
prob.model.add_design_var('wing.twist_cp', lower=-5., upper=10.)
prob.model.add_design_var('wing.thickness_cp', lower=0.001, upper=0.01, scaler=1e3)
prob.model.add_design_var('wing.sweep', lower=10., upper=30.)
prob.model.add_constraint('AS_point_0.wing_perf.failure', upper=0.)
prob.model.add_constraint('AS_point_0.wing_perf.thickness_intersects', upper=0.)

# Add design variables, constraisnt, and objective on the problem
prob.model.add_design_var('alpha', lower=-10., upper=10.)
prob.model.add_constraint('AS_point_0.L_equals_W', equals=0.)
prob.model.add_constraint('AS_point_0.CM', equals=0.)
prob.model.add_constraint('wing.twist_cp', lower=np.array([-1e20, -1e20, 5.]), upper=np.array([1e20, 1e20, 5.]))

prob.model.add_objective('AS_point_0.fuelburn', scaler=.1)

# Set up the problem
prob.setup(check=True)

# from openmdao.api import view_model
# view_model(prob)

prob.run_driver()