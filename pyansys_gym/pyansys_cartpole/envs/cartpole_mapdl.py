#!/usr/bin/env python
# coding: utf-8

# # CartPole
# This is a version of CartPole converted from Jonathan Glanville's original 2018 work with Google DeepMind  

# <img src="cart_pole_diagram.jpg" alt="Drawing" style="width: 800px;"/>

# The figure above shows the pole and its initial location, angle and length of the track
# It is defined by the following parameters:
#  * pole.radius
#  * pole.length
#  * theta
#  * start_pos
#  * track_length
#
# The pole is driven by a motor represented by a mass element to give it inertia.
# This elements node is connected to the bottom of the pole using a constraint equation, linking the
# x displacement with the rotation of the motor. A contac178 element, with cylindrical gap,
# is used to connect the pole "pilot" node with the pole so that a gap can open and close 
# when the cart changes direction. A mass element is added to the bottom of the pole to 
# represent the cart.
# 
#      MOTOR                   CE                            gap               CART MASS
#   node / mass element  ----------------- pilot node <---contact178 ----> bottom of pole 
#      node 12                              node 13                           node 1
# 
# figure 2 - Connection of the pole to the motor
# 
# additional parameters are:
#  * motor_izz - the moment of inertia of the motor
#  * cart_mass - the mass of the cart
#  * gear_ratio  - dist traverse / motor revolutions  (m/rad)
#  * gap_size  - the slack in the motor-cart connection (m)
# 
# This script has two further parameters of
#  * elaspe  - the time to run the simulation for 
#  * deltaT  - the timestep of the simulation				
# 
# And the material parameters:
#  * Youngs  - Young's Modulus (Pa)
#  * Poisson - Poisson's ratio
#  * Density - Density (kg m-3)
# 
# JPG
# 10/11/2018
# this code is for demonstration purposes and is not controlled
# feel free to refer questions to jon.glanville@ansys.com
from collections import namedtuple
import numpy as np
import re
import ansys
# mapdl = ansys.Mapdl()


class CartPoleMapdl:
    get_re = re.compile(r'VALUE\s*=\s*(\S+)')

    def __init__(self, **kwargs):
        self._preferences = kwargs
        if kwargs is None:
            self._preferences = {}

        # simulation parameters
        self._mapdl = mapdl

        self._cart_pos_0 = self._preferences.get('cart_pos_0', 0.)
        self._theta_deg_hint = self._preferences.get('_theta_deg_hint', 2.)
        self._theta_deg_max = self._preferences.get('theta_deg_max', 12.)
        self._start_pos = self._preferences.get('start_pos', 1.)
        self._track_length = self._preferences.get('track_length', 2.)

        self._reset_state()

        Pole = namedtuple('Pole', ['radius', 'length'])
        self._pole = Pole(12.7e-3, self._preferences.get('pole_length', 1.))

        self._youngs = 20.e9           # Polycarbonate
        self._poisson = 0.34
        self._density = 1220.

        self._cart_mass = self._preferences.get('cart_mass', .2)

        self._motor_izz = 0.01
        self._gap_size = 1e-3
        self._gear_ratio = 1.5e-3 * np.degrees(1)   # 1.5e-3 was per mm/degree here adjusted for radians

        self._max_time_points = self._preferences.get('max_time_points', 200)
        self._D_time = self._preferences.get('D_time', .05)
        self._d_time = self._preferences.get('d_time', .05)
        self._d_time_min = self._d_time/1000.
        self._d_time_max = self._d_time

        self._n_beam_elements = 10

        self._force_mag = self._preferences.get('force_mag', .1)
        self._force_dir = 1.

        self._elements = {'joint': 11}
        self._nodes = {'beam_begin': 1}
        self._nodes['beam_end'] = self._nodes['beam_begin'] + self._n_beam_elements
        self._nodes['motor'] = self._nodes['beam_end'] + 1
        self._nodes['cart_pilot'] = self._nodes['motor'] + 1

        # reuse for material, type, real, sec MAPDL sets
        self._ids = {
            'pole': 1,
            'joint': 2,
            'motor': 3,
            'gap': 4,
            'cart': 5}

    def _reset_state(self):
        self._finished = False
        self._cart_pos = self._cart_pos_0
        self._cart_velocity = 0.
        self._theta_deg = self._theta_deg_0 = np.random.uniform(-self._theta_deg_hint, self._theta_deg_hint)
        self._pole_velocity = 0.
        self._cur_time_point = 0
        self._reported_time_point = 0

    def reset(self):
        self._reset_state()

        # build the model
        mapdl.clear()
        mapdl.finish()
        mapdl.prep7()

        self._build_pole()
        self._build_pole_joint_to_ground()
        self._build_motor_mass_point()
        self._build_cart_pilot_node()
        self._build_cylindrical_gap()
        self._build_cart_mass_point()

        mapdl.slashsolu()
        mapdl.outres('erase')
        mapdl.outres('all', 'none')
        mapdl.outres('nsol', 'all')
        mapdl.outres('rsol', 'all')
        mapdl.outres('esol', 'all')
        mapdl.outres('v', 'all')
        mapdl.outres('a', 'all')
        mapdl.antype('trans')

        mapdl.nlgeom('on')
        mapdl.timint('on')
        mapdl.tintp(.05)
        mapdl.dmprat(.05)
        mapdl.autots('on')
        mapdl.deltim(self._d_time, self._d_time_min, self._d_time_max, 'off')

    def _build_pole(self):
        # ## pole
        mapdl.mp('EX', self._ids['pole'], self._youngs)            # Young's modulus in Pa
        mapdl.mp('NUXY', self._ids['pole'], self._poisson)         # Poisson's ratio
        mapdl.mp('DENS', self._ids['pole'], self._density)         # Density

        mapdl.sectype(self._ids['pole'], 'BEAM', 'CSOLID')
        mapdl.secdata(self._pole.radius)

        mapdl.n(self._nodes['beam_begin'])
        mapdl.n(self._nodes['beam_end'],
                self._pole.length * np.sin(np.radians(self._theta_deg)),
                self._pole.length * np.cos(np.radians(self._theta_deg)), 0)
        mapdl.fill(self._nodes['beam_begin'], self._nodes['beam_end'])

        mapdl.mat(self._ids['pole'])
        mapdl.real(self._ids['pole'])
        mapdl.type(self._ids['pole'])
        mapdl.secnum(self._ids['pole'])
        mapdl.et(self._ids['pole'], 188)                          # beam elements
        mapdl.e(self._nodes['beam_begin'], self._nodes['beam_begin'] + 1)   # create a beam element
        mapdl.egen(self._n_beam_elements, 1, -1)              # reproduce the pattern of the last element

    def _build_pole_joint_to_ground(self):
        # ## pole joint (body-to-ground)

        # define a coordinate system to represent the revolute joint at the bottom of the beam
        # it is aligned with the global origin
        csys_joint = 12
        cartesian_type = 0
        mapdl.local(csys_joint, cartesian_type, 0., 0., 0., 0., 0., 0.)
        mapdl.csys(0)       # reactivate the default coordinate system

        # create a joint from the beam to the ground
        mapdl.et(self._ids['joint'], 184)                   # multipoint constraint element type
        mapdl.keyopt(self._ids['joint'], 1, 16)             # mpc['element behavior'] = general joint
        mapdl.sectype(self._ids['joint'], 'joint', 'gene', 'track')
        mapdl.secjoint('', csys_joint, csys_joint)    # ids of the local cs for nodes i and j, respectively of the joint
        mapdl.secjoint('rdof', 2, 3, 4, 5)                  # DOFs to fix for the joint UY, YZ, ROTX, ROTY
        mapdl.secstop(1, -self._start_pos,
                      self._track_length - self._start_pos)     # provide stops to dof 1 (UX) at the ends of the track
        mapdl.mat(self._ids['joint'])
        mapdl.real(self._ids['joint'])
        mapdl.type(self._ids['joint'])
        mapdl.secnum(self._ids['joint'])
        mapdl.e(None, self._nodes['beam_begin'])            # create a joint element from ground (None) to beam

    def _build_motor_mass_point(self):
        # ## motor mass point
        # create a mass point for the motor, positioned at -startpos

        mapdl.n(self._nodes['motor'], -self._start_pos, 0, 0)
        mapdl.et(self._ids['motor'], 21)                # structural mass element type
        mapdl.r(self._ids['motor'])
        mapdl.rmodif(self._ids['motor'], 6, self._motor_izz)  # set the 6th real constant, i.e., rotary inertia about z
        mapdl.mat(self._ids['motor'])
        mapdl.real(self._ids['motor'])
        mapdl.type(self._ids['motor'])
        mapdl.secnum(self._ids['motor'])
        mapdl.e(self._nodes['motor'])
        mapdl.d(self._nodes['motor'], 'all', 0)                   # fix all DOFs on the motor, except rotation about z
        mapdl.ddele(self._nodes['motor'], 'rotz')

    def _build_cart_pilot_node(self):
        # ## cart pilot node
        #
        # constrain this node to the rotation of the "motor" (node 12)
        # $$ ^{13}u_X = \text{gear_ratio} \cdot ^{12}\theta_Z $$
        #
        # $$0 = -^{13}u_X + \text{gear_ratio} \cdot ^{12}\theta_Z $$

        mapdl.n(self._nodes['cart_pilot'], 0., 0., 0.)                  # create a cart with only UX dof free.
        mapdl.d(self._nodes['cart_pilot'], 'all', 0)
        mapdl.ddele(self._nodes['cart_pilot'], 'ux')

        ce_gear_ratio = 1
        const_term = 0
        mapdl.ce(ce_gear_ratio, const_term,
                 self._nodes['cart_pilot'], 'ux', -1.,
                 self._nodes['motor'], 'rotz', self._gear_ratio)

    def _build_cylindrical_gap(self):
        # ## cylindrical gap
        # between the bottom of the pole and the pilot node. i.e. non-perfect connection
        mapdl.et(self._ids['gap'], 178)
        mapdl.keyopt(self._ids['gap'], 1, 1)                  # cylindrical gap
        mapdl.keyopt(self._ids['gap'], 2, 1)                  # contact algorithm: pure penalty
        mapdl.keyopt(self._ids['gap'], 3, 0)                  # weak springs: none
        mapdl.keyopt(self._ids['gap'], 10, 0)                 # contact type: standard (like frictionless contact)
        mapdl.keyopt(self._ids['gap'], 12, 0)                 # contact status: do not print contact status
        mapdl.r(self._ids['gap'], -1.e9, self._gap_size)      # stiffness and gap size
        mapdl.rmodif(self._ids['gap'],
                     6, 0.0, 0.0, 1.0)      # real constants 6, 7, 8: direction cosines of the cylinder axis (normal z)
        mapdl.rmodif(self._ids['gap'],
                     12, 0.)                # real constant 12: CV1 (damping coefficient)
        mapdl.type(self._ids['gap'])
        mapdl.real(self._ids['gap'])
        mapdl.mat(self._ids['gap'])
        mapdl.secnum(self._ids['gap'])
        mapdl.e(self._nodes['beam_begin'], self._nodes['cart_pilot'])   # node 13 is inside the gap of node 1

    def _build_cart_mass_point(self):
        # ## mass point for the cart

        mapdl.et(self._ids['cart'], 21)
        mapdl.keyopt(self._ids['cart'], 3, 2)                 # rotary intertia options: 3-D mass without rotary inertia
        mapdl.r(self._ids['cart'], self._cart_mass)           # total mass (when keyopt(3) = 2, as is the case)
        mapdl.type(self._ids['cart'])
        mapdl.real(self._ids['cart'])
        mapdl.mat(self._ids['cart'])
        mapdl.secnum(self._ids['cart'])
        mapdl.e(self._nodes['beam_begin'])

    def act(self, force_dir):
        self._force_dir = force_dir

    def step(self, force_dir):
        if self._finished:
            return

        self._cur_time_point += 1
        self._force_dir = force_dir
        mapdl.acel(0., 9.80665, 0.)         # the global cs is accelerated upwards (so a gravity is felt downwards)
        mapdl.f(self._nodes['motor'], 'mz', self._force_dir * self._force_mag)
        mapdl.time(self._cur_time_point * self._D_time)
        mapdl.solve()
        self._query_state()

    def _query_state(self):
        def get_result(get_txt):
            try:
                return float(CartPoleMapdl.get_re.findall(get_txt)[0])
            except ValueError:
                return np.nan

        self._cart_pos = self._cart_pos_0 + get_result(mapdl.get('dummy', 'node', 1, 'u', 'x'))
        self._cart_velocity = get_result(mapdl.get('dummy', 'node', 1, 'v', 'x'))
        self._theta_deg = self._theta_deg_0 - np.degrees(get_result(mapdl.get('dummy', 'elem', 11, 'SMISC', 66)))
        self._pole_velocity = get_result(mapdl.get('dummy', 'node', 10, 'v', 'sum'))
        self._reported_time_point = int(np.round(get_result(mapdl.get('dummy', 'active', 0, 'SET', 'LSTP'))))
        if self._cur_time_point != self._reported_time_point or \
                self._cur_time_point > self._max_time_points:
            self._finished = True

        if np.abs(self._theta_deg) > self._theta_deg_max:
            self._finished = True

    def get_state(self):
        return np.array([self._cart_pos, self._cart_velocity, self._theta_deg, self._pole_velocity],
                        dtype=np.double)

    def is_over(self):
        return self._finished
