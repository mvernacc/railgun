import numpy as np
from scipy import constants
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def contact_resistance(params):
    """Resistance of one contact between the armature and a rail.
    Bullshit model from https://pdxscholar.library.pdx.edu/cgi/viewcontent.cgi?article=1324&context=open_access_etds
    pdf page 19"""
    # Model constant [units: ohm N]
    k = 0.001 * 4.448
    # k *= 0.1
    return k / contact_force(params)


def rail_resistance(x_a, params):
    """Resistance of one rail, with armature at position x_a."""
    return x_a * params['rail resistivity'] / (
        params['rail height'] * params['rail width'])


def armature_resistance(params):
    return params['rail gap'] * params['armature resistivity'] / (
        params['rail height'] * params['armature length'])


def loop_resistance(x_a, params):
    return (
        2 * rail_resistance(x_a, params)
        + 2 * contact_resistance(params)
        + armature_resistance(params)
        + params['capacitor resistance']
        )


def contact_force(params):
    """Contact force between rail and armature, due to wedge angle and gravity.
    Returns:
        force [units: N]
    """
    return params['armature mass'] * constants.g / (
        2 * np.sin(params['contact angle']))

def armature_force_rails(I, params):
    """Force on the armature due to the rail's B field.
    Returns:
        force [units: N]
    """
    return constants.mu_0 / np.pi * I**2 * np.arctanh(
        params['rail gap'] / (params['rail width'] + params['rail gap']))


def armature_force_perm(I, params):
    """Force on the armature due to permanent magnets.
    Returns:
        force [units: N]"""
    return I * params['rail gap']  * params['perm magnet field']


def event_reached_end_of_rail(t, state, params):
    """Event function for solve_ivp.
    Crosses 0 when armature reached the end of the rails."""
    x_a = state[1]
    return params['rail length'] - x_a


def dynamics(t, state, params):
    V = state[0]
    x_a = state[1]
    v_a = state[2]

    # Loop resistance [units: ohm]
    R = loop_resistance(x_a, params)
    # Current thru the rails and armature [units: A]
    I = V / R
    # Capacitor discharge rate [units: V s**-1]
    dV_dt = -I / params['capacitance']

    # Force on the armature [units: N]
    force = (armature_force_rails(I, params)
             + armature_force_perm(I, params))
    # Armature acceleration [units: m s**-2]
    dv_a_dt = force / params['armature mass']
    dx_a_dt = v_a

    return np.array([
        dV_dt, dx_a_dt, dv_a_dt])



def main():
    params = {
        'rail gap': 6.35e-3, # units: m
        'rail width': 6.35e-3, # units: m
        'rail height': 6.35e-3, # units: m
        'rail length': 150e-3, # units: m
        'armature length': 5e-3, # units: m
        'contact angle': np.deg2rad(5.),
        # For a stainless steel armature
        'armature resistivity': 6.9e-7, # units: ohm m
        'armature density': 8000., # units: kg m^-3
        # For copper rails
        'rail resistivity': 1.68e-8, # units: ohm m
        # For KEMET ALS7(1)(2)514NS040
        'capacitance': 0.51, # units: F
        'capacitor resistance': 7e-3, # units: ohm
        'perm magnet field': 0.4, # units: T
    }
    params['armature mass'] = (
        params['armature length'] * params['rail height'] * params['rail gap']
        * params['armature density'])
    print('Armature mass = {:.3f} g'.format(
        1e3 * params['armature mass']))
    print('Resistances:')
    print('\t2x contacts = {:.0f} mOhm'.format(
        2e3 * contact_resistance(params)))
    print('\t2x rails at 100 mm = {:.3f} mOhm'.format(
        2e3 * rail_resistance(0.1, params)))
    print('\tArmature = {:.3f} mOhm'.format(
        1e3 * armature_resistance(params)))
    print('\tCapacitor ESR = {:.0f} mOhm'.format(
        1e3 * params['capacitor resistance']))

    # Initial state
    V_init = 40.
    state_init = np.array([V_init, 0., 0.])

    # Time span to integrate
    t_span = (0., 100e-3)
    # Termination event
    event_reached_end_of_rail.terminal = True

    result = solve_ivp(
        fun=dynamics, t_span=t_span, y0=state_init,
        args=(params,), method='Radau',
        events=event_reached_end_of_rail)
    t = result.t
    V = result.y[0, :]
    x_a = result.y[1, :]
    v_a = result.y[2, :]
    R = loop_resistance(x_a, params)
    I = V / R

    kinetic_energy = 0.5 * params['armature mass'] * v_a[-1]**2
    print('Armature kinetic energy = {:.1f} J'.format(
        kinetic_energy))
    energy_used = 0.5 * params['capacitance'] * (
        V[0]**2 - V[-1]**2)
    print('Energy from capacitor = {:.1f} J'.format(
        energy_used))
    print('Efficiency = {:.3f} percent'.format(
        100 * kinetic_energy / energy_used))

    # Plot results
    fig, axes = plt.subplots(
        figsize=(6, 10),
        nrows=4, ncols=1, sharex=True)
    ax_I, ax_volt, ax_x, ax_v = axes

    ax_I.plot(1e3 * t, I)
    ax_I.set_ylabel('$I$ [A]')

    ax_volt.plot(1e3 * t, V)
    ax_volt.set_ylabel('$V$ [V]')

    ax_x.plot(1e3 * t, 1e3 * x_a, marker='.')
    ax_x.set_ylabel('$x_a$ [mm]')

    ax_v.plot(1e3 * t, v_a)
    ax_v.set_ylabel('$v_a$ [m s$^{-1}$]')
    ax_v.set_xlabel('$t$ [ms]')

    plt.suptitle(
        'Railgun simulation'
        + '\n$C =$ {:.2f} F, $B_{{permanent}}$ = {:.2f} T'.format(
            params['capacitance'], params['perm magnet field']))

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()


if __name__ == '__main__':
    main()
