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
 
# JPG
# 10/11/2018
# this code is for demonstration purposes and is not controlled
# feel free to refer questions to jon.glanville@ansys.com
from collections import namedtuple
import numpy as np
import re
from ansys.mapdl.core import launch_mapdl
mapdl = launch_mapdl(loglevel='ERROR', verbose=False)


class CartPoleMapdlSimple:
    get_re = re.compile(r'VALUE\s*=\s*(\S+)')

    def __init__(self, **kwargs):
        self._preferences = kwargs
        if kwargs is None:
            self._preferences = {}

        # simulation parameters
        self._mapdl = mapdl

        self._cart_pos_hint = self._preferences.get('cart_pos_hint', .05)
        self._cart_velocity_hint = self._preferences.get('cart_velocity_hint', .05)
        self._theta_deg_hint = self._preferences.get('_theta_deg_hint', np.degrees(.05))
        self._theta_deg_max = self._preferences.get('theta_deg_max', 12.)
        self._track_length = self._preferences.get('track_length', 2 * 2.4)

        self._reset_state()

        Pole = namedtuple('Pole', ['radius', 'length'])
        self._pole = Pole(self._preferences.get('pole_radius', .01), self._preferences.get('pole_length', 1.))

        self._youngs = 20.e9           # Polycarbonate
        self._poisson = 0.34
        self._density = 1000.

        self._cart_mass = self._preferences.get('cart_mass', 1.)

        self._max_time_points = self._preferences.get('max_time_points', 200)
        self._D_time = self._preferences.get('D_time', .02)
        self._d_time = self._preferences.get('d_time', .02)
        self._d_time_min = self._d_time/1000.
        self._d_time_max = self._d_time

        self._n_beam_elements = 10

        self._force_mag = self._preferences.get('force_mag', 10.)
        self._force_dir = 1.

        self._elements = {'joint': 11}
        self._nodes = {'beam_begin': 1}
        self._nodes['beam_end'] = self._nodes['beam_begin'] + self._n_beam_elements

        # reuse for material, type, real, sec MAPDL sets
        self._ids = {
            'pole': 1,
            'joint': 2,
            'motor': 3,
            'gap': 4,
            'cart': 5}

    def _reset_state(self):
        self._finished = False
        self._cart_pos = self._cart_pos_0 = np.random.uniform(-self._cart_pos_hint, self._cart_pos_hint)
        self._cart_velocity = self._cart_velocity_0 = \
            np.random.uniform(-self._cart_velocity_hint, self._cart_velocity_hint)
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

        mapdl.ic(self._nodes['beam_begin'], 'vx', self._cart_velocity_0)    # initial conditions

    def _build_pole(self):
        # ## pole
        mapdl.mp('EX', self._ids['pole'], self._youngs)            # Young's modulus in Pa
        mapdl.mp('NUXY', self._ids['pole'], self._poisson)         # Poisson's ratio
        mapdl.mp('DENS', self._ids['pole'], self._density)         # Density

        mapdl.sectype(self._ids['pole'], 'BEAM', 'RECT')
        mapdl.secdata(self._pole.radius, self._pole.radius)

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
        mapdl.secstop(1, -self._track_length/2.,
                      self._track_length / 2.)              # provide stops to dof 1 (UX) at the ends of the track
        mapdl.mat(self._ids['joint'])
        mapdl.real(self._ids['joint'])
        mapdl.type(self._ids['joint'])
        mapdl.secnum(self._ids['joint'])
        mapdl.e(None, self._nodes['beam_begin'])            # create a joint element from ground (None) to beam

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
        mapdl.f(self._nodes['beam_begin'], 'fx', self._force_dir * self._force_mag)
        mapdl.time(self._cur_time_point * self._D_time)
        mapdl.solve()
        self._query_state()

    def _query_state(self):
        self._cart_pos = self._cart_pos_0 + mapdl.get_value('node', 1, 'u', 'x')
        self._cart_velocity = mapdl.get_value('node', 1, 'v', 'x')
        self._theta_deg = self._theta_deg_0 - np.degrees(mapdl.get_value('elem', 11, 'SMISC', 66))
        self._pole_velocity = mapdl.get_value('node', 10, 'v', 'sum')
        self._reported_time_point = int(np.round(mapdl.get_value('active', 0, 'SET', 'LSTP')))

        if self._cur_time_point != self._reported_time_point or \
                self._cur_time_point > self._max_time_points:
            self._finished = True

        if np.abs(self._theta_deg) > self._theta_deg_max:
            self._finished = True

    def get_state(self):
        return np.array([self._cart_pos, self._cart_velocity, self._theta_deg, self._pole_velocity],
                        dtype=np.float)

    def is_over(self):
        return self._finished
