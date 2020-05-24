import numpy as np
from scipy import constants
import matplotlib.pyplot as plt


def contact_resistance(load):
    """Resistance of one contact between the armature and a rail.
    Bullshit model from https://pdxscholar.library.pdx.edu/cgi/viewcontent.cgi?article=1324&context=open_access_etds
    pdf page 19"""
    # Model constant [units: ohm N]
    k = 0.001 * 4.448
    # k *= 0.1
    return k / load


def main():
    # Load data
    data = np.genfromtxt(
        'data/penny_contact_resistance_vs_load.csv',
        delimiter=',',
        skip_header=1)
    # Load pressing armature onto rails [units: newton]
    load = data[:, 0] * constants.g * 1e-3
    # Rail-to-rail resistance through 2x contacts and armature [units: ohm]
    resistance = data[:, 1] * 1e-3

    load_model = np.linspace(0.1, 8, 100)
    resistance_model = 2 * contact_resistance(load_model)

    plt.plot(
        load, 1e3 * resistance,
        marker='x', color='black', linestyle='none',
        label='measurements')
    plt.plot(
        load_model, 1e3 * resistance_model,
        color='tab:blue',
        label='Model')
    plt.xlabel('Load pressing armature onto rails [N]')
    plt.ylabel('Resistance [mOhm]')
    plt.legend()


if __name__ == '__main__':
    main()
    plt.show()
