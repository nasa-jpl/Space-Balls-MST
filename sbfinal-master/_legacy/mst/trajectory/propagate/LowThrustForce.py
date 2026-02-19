"""

===================
low_thrust_force.py
===================

Class defining a low-thrust Monte.PyForce for applying constant low-thrust
to a trajectory.

"""

import Monte as M
import mpy.units as units

import weakref

import numpy as np

#from incus.config import CONFIG


ENTER_ECLIPSE_INTERVAL = M.ShadowEvent.PointType.PENUMBRA_ENTRY
EXIT_ECLIPSE_INTERVAL = M.ShadowEvent.PointType.PENUMBRA_EXIT

NOT_ECLIPSE = M.Shadow.ShadowStatus.IN_CLEAR


class EclipseInterruptHandler:
    def __init__(self, body, force):
        self.name = "Eclipse Event Handler"
        self.body = body
        self.force = force

    def handleTime(self, forward, epoch, delta, infoEvents):
        return M.IntegHandler.RESTART

    def handleEvent(self, forward, event, delta, infoEvents):
        if "entry" in event.type():
            self.force.thrust_on = 0
        else:
            self.force.thrust_on = 1

        # Restart the the integration.
        return M.IntegHandler.CONTINUE


class LowThrustForce:
    def __init__(self, boa, body, thrust, eclipse=True):
        self.boa = boa

        # The body this burn is for.
        self.body = body

        # Traj query
        self.query = M.TrajQuery(
            boa, self.body, "Earth", "EME2000" #CONFIG["CENTRAL_BODY"], CONFIG["INERTIAL_FRAME"]
        )

        # mass
        self.mass = M.MassBoa.read(boa, self.body)

        # flag for if we need to deal with eclipses
        self.eclipse = eclipse

        # Shadow status
        self.shadow = M.Shadow(boa, "Earth", body) #CONFIG["CENTRAL_BODY"], body)

        # continuous low thrust
        self.thrust = thrust

        # previous derivative
        self.prev_accel = None

        # flag if thrust is on
        self.thrust_on = None

    def name(self):
        return f"Low Thrust Eclipse Force {self.body}"

    def isActive(self):
        return True

    def initializeForce(self, forward, epoch, isReset, increment):
        if self.eclipse == False:
            self.thrust_on = 1

        elif self.in_eclipse(epoch):
            self.thrust_on = 0
        else:
            self.thrust_on = 1

    def in_eclipse(self, epoch):
        shadow_status = self.shadow.status(epoch)
        if shadow_status != NOT_ECLIPSE:
            return False
        else:
            return True

    def accel(self, epoch):
        # Get the position vector as a Unit3Vec
        uvel = self.query.state(epoch).vel().unit()
        mass = self.mass.mass(epoch)
        accel = (self.thrust / mass * uvel).value()

        # Convert to Monte std units in a Dbl3Vec for the force.
        return np.array(accel)

    def resetIntegSetup(self, integSetup, resetModels):
        integState = integSetup.findState(self.body)
        self.index = integState.index()

    def compute(self, epoch, deriv, smooth=0.1):
        # Compute the acceleration.
        if self.index != -1:
            # check eclipse status
            accel = self.thrust_on * self.accel(epoch)

            # Normally, state integration uses Cartesian elements,
            # and there are always three 2nd order equations.
            for i, component in zip(range(3), accel):
                # Set the acceleration in the deriv array to be returned.
                # This example returns 0 since no acceleration is computed.
                deriv[self.index + i] = component

    def getInterrupts(
        self, prop_name, forward, begin_epoch, end_epoch, search_step=10 * units.minute
    ):
        # find shadow events
        enter_eclipse_spec = M.ShadowEvent(
            self.boa, "Earth", self.body, ENTER_ECLIPSE_INTERVAL
        )
        exit_eclipse_spec = M.ShadowEvent(
            self.boa, "Earth", self.body, EXIT_ECLIPSE_INTERVAL
        )

        # event handler
        handler1 = M.PyIntegHandler(EclipseInterruptHandler(self.body, self))
        handler2 = M.PyIntegHandler(EclipseInterruptHandler(self.body, self))

        # two interrupts for enter/exit
        enter_eclipse_interrupt = M.IntegEventInt(enter_eclipse_spec, handler1, True)
        exit_eclipse_interrupt = M.IntegEventInt(exit_eclipse_spec, handler2, True)

        timeInt = []
        if self.eclipse:
            eventInt = [enter_eclipse_interrupt, exit_eclipse_interrupt]
        else:
            eventInt = []
        frames = []
        resetTimes = []

        return timeInt, eventInt, frames, resetTimes