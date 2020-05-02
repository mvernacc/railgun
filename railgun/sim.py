"""Railgun simulation."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy import constants
import matplotlib.pyplot as plt


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


def main():
    # Width between the rails [units: m]
    w = 5e-3
    # Width of the rails and armature [units: m]
    a = 1e-3
    # Current [units: A]
    I = 10e3
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
    main()
