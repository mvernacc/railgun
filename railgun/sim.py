"""Railgun simulation."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy import constants
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Resisivity of copper [units: ohm m]
res_Cu = 1.7e-8
# Resisivity of graphite [units: ohm m]
res_graphite = 5e-5


def inductance_rectuangular_loop(w, h, a):
    """Inductance of a single rectangular loop.
    Formula from https://cecas.clemson.edu/cvel/emc/calculators/Inductance_Calculator/rectgl.html
    """
    sqrt_term = (h**2 + w**2)**0.5
    return constants.mu_0 / np.pi * (
        -2 * (w + h) + 2 * sqrt_term
        - h * jnp.log((h + sqrt_term) / w)
        - w * jnp.log((w + sqrt_term) / h)
        + h * jnp.log(2 * h / a)
        + w * jnp.log(2 * w / a)
        )


def resistance_loop(x_a, a, w):
    # Resistance of each rail
    R_rail = res_Cu * x_a / a**2
    # Resistance of the armature
    R_arm = res_graphite / w
    return 2 * R_rail + R_arm + 0.005


def dynamics(t, state, params, dL_dx_a):
    I = state[0]
    V = state[1]
    x_a = state[2]
    v_a = state[3]

    # Inductance [units: H]
    L = inductance_rectuangular_loop(params['w'], x_a, params['a'])
    L = float(L)
    # L = 20e-9 * (1 + x_a / 20e-3)
    # Resisitance [units: ohm]
    R = resistance_loop(x_a, params['a'], params['w'])

    dI_dt = 1 / L * (V -  I * R)
    dV_dt = -I / params['C']

    # Force on the armature [units: N]
    F_a = 0.5 * I**2 * float(dL_dx_a(x_a))
    # F_a = 0.1

    dstate_dt = np.array([
        dI_dt, dV_dt, v_a, F_a / params['m_a']])
    return dstate_dt


def main2():
    params = {
        # Width between the rails [units: m]
        'w': 5e-3,
        # Width of the rails and armature [units: m]
        'a': 2e-3,
        # Mass of the armature [units: kg]
        'm_a': 0.6e-3,
        # Capacitor capacitance [units: F]
        'C': 0.5
        }
    dL_dx_a = jax.grad(
        lambda x_a: inductance_rectuangular_loop(
            params['w'], x_a, params['a']))
    # Initial capacitor voltage [units: volt]
    V_init = 40.
    x_a_init = 10e-3
    R_init = resistance_loop(x_a_init, params['a'], params['w'])
    print('R_init = {:.3f} ohm'.format(R_init))
    I_steady = V_init / R_init
    # Time span to integrate
    t_span = (0., 20e-3)

    state_init = np.array([I_steady, V_init, x_a_init, 0.])
    result = solve_ivp(
        fun=dynamics, t_span=t_span, y0=state_init,
        args=(params, dL_dx_a), method='Radau')
    t = result.t
    I = result.y[0, :]
    V = result.y[1, :]
    x_a = result.y[2, :]
    v_a = result.y[3, :]

    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True)
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

    plt.tight_layout()
    plt.show()


def main():
    # Width between the rails [units: m]
    w = 5e-3
    # Width of the rails and armature [units: m]
    a = 2e-3
    # Current [units: A]
    I = 1e3
    # Armature position [units: m]
    x_a = np.linspace(10e-3, 200e-3)

    # Inductance [units: H]
    L = inductance_rectuangular_loop(w, x_a, a)

    # Derivative of inductance w.r.t. armature position
    dL_dx_a = jax.grad(
        lambda x_a: inductance_rectuangular_loop(w, x_a, a))

    # Force on the armature [units: N]
    F_a = np.zeros(len(x_a))
    for i in range(len(x_a)):
        F_a[i] = 0.5 * I**2 * dL_dx_a(x_a[i])

    # External field [units: T]
    B_ext = 0.05
    F_ext = w * I * B_ext
    print('F from B_ext = {:.3f} N'.format(F_ext))

    # Work done on the armature [units: J]
    work_a = np.trapz(F_a, x_a)

    print('L = {:.3f} nH, F_a = {:.3f} N'.format(
        1e9 * L[0], F_a[0]))
    print('Total work on armature = {:.3f} J'.format(work_a))

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax_L, ax_F = axes
    ax_L.plot(1e3 * x_a, 1e9 * L)
    ax_L.set_ylabel('$L$ [nH]')

    ax_F.plot(1e3 * x_a, F_a)
    ax_F.set_ylabel('$F_a$ [N]')
    ax_F.set_xlabel('$x_a$ [mm]')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # main()
    main2()
