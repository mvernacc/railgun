"""Unit tests"""
import pytest
from pytest import approx
import sim


class TestInductanceRectangularLoop:
    def test_1_1(self):
        """Inductance of a 1 x 1 cm loop."""
        a = 1e-3  # [units: meter]
        w = 1e-2  # [units: meter]
        h = 1e-2  # [units: meter]
        # Answer from https://cecas.clemson.edu/cvel/emc/calculators/Inductance_Calculator/rectgl.html
        # [units: henry]
        L_correct = 12.23e-9

        L = sim.inductance_rectuangular_loop(w, h, a)
        L = float(L)

        assert L == approx(L_correct, abs=0.01e-9)

    def test_1_2(self):
        """Inductance of a 1 x 2 cm loop."""
        a = 1e-3  # [units: meter]
        w = 2e-2  # [units: meter]
        h = 1e-2  # [units: meter]
        # Answer from https://cecas.clemson.edu/cvel/emc/calculators/Inductance_Calculator/rectgl.html
        # [units: henry]
        L_correct = 21.91e-9

        L = sim.inductance_rectuangular_loop(w, h, a)
        L = float(L)

        assert L == approx(L_correct, abs=0.01e-9)
