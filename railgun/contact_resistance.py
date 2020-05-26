import numpy as np
from scipy import constants
import matplotlib.pyplot as plt

import lmfit

def contact_resistance(load):
    """Resistance of one contact between the armature and a rail.
    Fit to my experiments iwth a penny on 2020-05-23."""
    # Reference load [units: newton]
    load_ref = 1.46
    # Reference resistance [units: ohm]
    res_ref = 26.7e-3
    return res_ref * np.exp(-load / load_ref)


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


    # Fit the model
    model = lmfit.models.ExponentialModel()
    fit_result = model.fit(
        resistance / 2, x=load / 2,
        amplitude=30e-3, decay=2.)
    print(fit_result.fit_report())

    load_model = np.linspace(0.1, 8, 100)
    # Half the load on each contact, 2 contacts in series
    resistance_model = 2 * model.eval(
        x=load_model / 2, params=fit_result.params)

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
    plt.ylim([0, 100])
    plt.legend()


if __name__ == '__main__':
    main()
    plt.show()
