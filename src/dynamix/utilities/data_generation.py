"""
Data generation module for dynamical systems.

This module provides functions to generate synthetic time series data from
classical chaotic and periodic dynamical systems for training DynaMix models.

Implements all 34 3D dynamical systems from the Gilpin/dysts benchmark used
in the DynaMix paper training corpus (Hemmer & Durstewitz, NeurIPS 2025).

Training systems (34 total):
- Aizawa, AnishchenkoAstakhov, Arneodo, BelousovZhabotinsky
- Chen, Colpitts, ForcedBrusselator, ForcedVanDerPol
- GlycolyticOscillation, HastingsPowell, HenonHeiles, Hopfield
- KawczynskiStrizhak, LiuChen, Lorenz, LorenzBounded, LorenzStenflo
- LuChenCheng, NewtonLiepnik, NuclearQuadrupole, PiecewiseCircuit
- Qi, QiChen, RabinovichFabrikant, RikitakeDynamo, Rossler, Rucklidge
- ShimizuMorioka, SprottJerk, SprottLinzSprott, StickSlipOscillator
- SwingingAtwood, Thomas, Windmi
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager

# Global variables for multiprocessing progress tracking
_shared_counter = None
_shared_lock = None


@dataclass
class SystemConfig:
    """Configuration for a dynamical system."""

    name: str
    dim: int
    default_params: Dict[str, float]
    param_ranges: Dict[str, Tuple[float, float]]
    default_ic_range: Tuple[float, float]


# =============================================================================
# System configurations for all 34 training systems from the DynaMix paper
# Parameters and IC ranges from the dysts benchmark (Gilpin, 2021)
# =============================================================================

SYSTEM_CONFIGS = {
    # 1. Aizawa
    "aizawa": SystemConfig(
        name="Aizawa",
        dim=3,
        default_params={"a": 0.95, "b": 0.7, "c": 0.6, "d": 3.5, "e": 0.25, "f": 0.1},
        param_ranges={
            "a": (0.85, 1.05),
            "b": (0.6, 0.8),
            "c": (0.5, 0.7),
            "d": (3.0, 4.0),
            "e": (0.2, 0.3),
            "f": (0.05, 0.15),
        },
        default_ic_range=(-0.5, 0.5),
    ),
    # 2. AnishchenkoAstakhov
    "anishchenko_astakhov": SystemConfig(
        name="AnishchenkoAstakhov",
        dim=3,
        default_params={"mu": 1.2, "eta": 0.5},
        param_ranges={"mu": (1.0, 1.4), "eta": (0.4, 0.6)},
        default_ic_range=(-1.0, 1.0),
    ),
    # 3. Arneodo
    "arneodo": SystemConfig(
        name="Arneodo",
        dim=3,
        default_params={"a": -5.5, "b": 3.5, "c": -1.0},
        param_ranges={"a": (-6.0, -5.0), "b": (3.0, 4.0), "c": (-1.2, -0.8)},
        default_ic_range=(-0.5, 0.5),
    ),
    # 4. BelousovZhabotinsky
    "belousov_zhabotinsky": SystemConfig(
        name="BelousovZhabotinsky",
        dim=3,
        default_params={"eps1": 0.001, "eps2": 0.1, "q": 0.01, "f": 1.0},
        param_ranges={
            "eps1": (0.0005, 0.002),
            "eps2": (0.08, 0.12),
            "q": (0.008, 0.015),
            "f": (0.8, 1.2),
        },
        default_ic_range=(0.01, 0.5),
    ),
    # 5. Chen
    "chen": SystemConfig(
        name="Chen",
        dim=3,
        default_params={"a": 35.0, "b": 3.0, "c": 28.0},
        param_ranges={"a": (32.0, 38.0), "b": (2.5, 3.5), "c": (25.0, 31.0)},
        default_ic_range=(-10.0, 10.0),
    ),
    # 6. Chua circuit - piecewise-linear chaotic oscillator
    "chua": SystemConfig(
        name="Chua",
        dim=3,
        default_params={"alpha": 15.6, "beta": 28.0, "m0": -1.143, "m1": -0.714},
        param_ranges={
            "alpha": (14.0, 17.0),
            "beta": (25.0, 31.0),
            "m0": (-1.3, -1.0),
            "m1": (-0.8, -0.6),
        },
        default_ic_range=(-0.5, 0.5),
    ),
    # 7. Colpitts (simplified jerk-like formulation)
    "colpitts": SystemConfig(
        name="Colpitts",
        dim=3,
        default_params={"a": 4.0, "b": 1.0, "c": 0.07, "d": 0.2},
        param_ranges={
            "a": (3.5, 4.5),
            "b": (0.8, 1.2),
            "c": (0.05, 0.1),
            "d": (0.15, 0.25),
        },
        default_ic_range=(-0.5, 0.5),
    ),
    # 7. ForcedBrusselator
    "forced_brusselator": SystemConfig(
        name="ForcedBrusselator",
        dim=3,
        default_params={"a": 0.4, "b": 1.2, "f": 0.05, "omega": 0.81},
        param_ranges={
            "a": (0.35, 0.45),
            "b": (1.1, 1.3),
            "f": (0.03, 0.07),
            "omega": (0.75, 0.85),
        },
        default_ic_range=(0.1, 2.0),
    ),
    # 8. ForcedVanDerPol
    "forced_vanderpol": SystemConfig(
        name="ForcedVanDerPol",
        dim=3,
        default_params={"mu": 8.53, "a": 1.2, "omega": 0.63},
        param_ranges={
            "mu": (8.0, 9.0),
            "a": (1.0, 1.4),
            "omega": (0.58, 0.68),
        },
        default_ic_range=(-2.0, 2.0),
    ),
    # 9. GlycolyticOscillation
    "glycolytic_oscillation": SystemConfig(
        name="GlycolyticOscillation",
        dim=3,
        default_params={
            "a": 100.0,
            "b": 0.1,
            "c": 4.0,
            "d": 4.0,
            "e": 0.02,
            "f": 2.0,
            "g": 3.5,
        },
        param_ranges={
            "a": (90.0, 110.0),
            "b": (0.08, 0.12),
            "c": (3.5, 4.5),
            "d": (3.5, 4.5),
            "e": (0.015, 0.025),
            "f": (1.8, 2.2),
            "g": (3.2, 3.8),
        },
        default_ic_range=(0.1, 1.0),
    ),
    # 10. HastingsPowell
    "hastings_powell": SystemConfig(
        name="HastingsPowell",
        dim=3,
        default_params={
            "a1": 5.0,
            "a2": 0.1,
            "b1": 3.0,
            "b2": 2.0,
            "d1": 0.4,
            "d2": 0.01,
        },
        param_ranges={
            "a1": (4.5, 5.5),
            "a2": (0.08, 0.12),
            "b1": (2.8, 3.2),
            "b2": (1.8, 2.2),
            "d1": (0.35, 0.45),
            "d2": (0.008, 0.012),
        },
        default_ic_range=(0.1, 1.0),
    ),
    # 11. HenonHeiles
    "henon_heiles": SystemConfig(
        name="HenonHeiles",
        dim=4,  # 4D system: (x, y, px, py)
        default_params={"lam": 1.0},
        param_ranges={"lam": (0.8, 1.2)},
        default_ic_range=(-0.3, 0.3),
    ),
    # 12. Hopfield
    "hopfield": SystemConfig(
        name="Hopfield",
        dim=3,
        default_params={"a": 1.0, "b": 3.0, "c": 1.0, "d": 5.0, "e": 0.16},
        param_ranges={
            "a": (0.9, 1.1),
            "b": (2.8, 3.2),
            "c": (0.9, 1.1),
            "d": (4.5, 5.5),
            "e": (0.14, 0.18),
        },
        default_ic_range=(-0.5, 0.5),
    ),
    # 13. KawczynskiStrizhak
    "kawczynski_strizhak": SystemConfig(
        name="KawczynskiStrizhak",
        dim=3,
        default_params={"beta": 2.5, "gamma": 0.05, "kappa": 0.1, "mu": 0.9},
        param_ranges={
            "beta": (2.2, 2.8),
            "gamma": (0.04, 0.06),
            "kappa": (0.08, 0.12),
            "mu": (0.8, 1.0),
        },
        default_ic_range=(0.0, 1.0),
    ),
    # 14. LiuChen
    "liu_chen": SystemConfig(
        name="LiuChen",
        dim=3,
        default_params={"a": 0.4, "b": 12.0, "c": 5.0, "d": 1.0, "e": 1.0, "f": 4.0},
        param_ranges={
            "a": (0.35, 0.45),
            "b": (11.0, 13.0),
            "c": (4.5, 5.5),
            "d": (0.9, 1.1),
            "e": (0.9, 1.1),
            "f": (3.5, 4.5),
        },
        default_ic_range=(-3.0, 3.0),
    ),
    # 15. Lorenz
    "lorenz": SystemConfig(
        name="Lorenz",
        dim=3,
        default_params={"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
        param_ranges={
            "sigma": (9.0, 11.0),
            "rho": (25.0, 31.0),
            "beta": (2.5, 3.0),
        },
        default_ic_range=(-10.0, 10.0),
    ),
    # 16. LorenzBounded
    "lorenz_bounded": SystemConfig(
        name="LorenzBounded",
        dim=3,
        default_params={"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
        param_ranges={
            "sigma": (9.0, 11.0),
            "rho": (25.0, 31.0),
            "beta": (2.5, 3.0),
        },
        default_ic_range=(-10.0, 10.0),
    ),
    # 17. LorenzStenflo
    "lorenz_stenflo": SystemConfig(
        name="LorenzStenflo",
        dim=4,
        default_params={"a": 2.0, "b": 0.7, "c": 26.0, "d": 1.5},
        param_ranges={
            "a": (1.8, 2.2),
            "b": (0.6, 0.8),
            "c": (24.0, 28.0),
            "d": (1.3, 1.7),
        },
        default_ic_range=(-5.0, 5.0),
    ),
    # 18. LuChenCheng
    "lu_chen_cheng": SystemConfig(
        name="LuChenCheng",
        dim=3,
        default_params={"a": 36.0, "b": 3.0, "c": 20.0},
        param_ranges={"a": (33.0, 39.0), "b": (2.5, 3.5), "c": (18.0, 22.0)},
        default_ic_range=(-10.0, 10.0),
    ),
    # 19. NewtonLiepnik
    "newton_leipnik": SystemConfig(
        name="NewtonLiepnik",
        dim=3,
        default_params={"a": 0.4, "b": 0.175},
        param_ranges={"a": (0.35, 0.45), "b": (0.15, 0.2)},
        default_ic_range=(-0.5, 0.5),
    ),
    # 20. NuclearQuadrupole
    "nuclear_quadrupole": SystemConfig(
        name="NuclearQuadrupole",
        dim=3,
        default_params={"a": 1.0, "b": 0.55, "c": 0.2},
        param_ranges={"a": (0.9, 1.1), "b": (0.5, 0.6), "c": (0.15, 0.25)},
        default_ic_range=(-1.0, 1.0),
    ),
    # 21. PiecewiseCircuit
    "piecewise_circuit": SystemConfig(
        name="PiecewiseCircuit",
        dim=3,
        default_params={"a": 0.6, "b": 1.0},
        param_ranges={"a": (0.5, 0.7), "b": (0.9, 1.1)},
        default_ic_range=(-0.5, 0.5),
    ),
    # 22. Qi
    "qi": SystemConfig(
        name="Qi",
        dim=3,
        default_params={"a": 45.0, "b": 10.0, "c": 1.0, "d": 10.0},
        param_ranges={
            "a": (42.0, 48.0),
            "b": (9.0, 11.0),
            "c": (0.9, 1.1),
            "d": (9.0, 11.0),
        },
        default_ic_range=(-5.0, 5.0),
    ),
    # 23. QiChen (using stable Chen-like parameters)
    "qi_chen": SystemConfig(
        name="QiChen",
        dim=3,
        default_params={"a": 30.0, "b": 3.0, "c": 28.0},
        param_ranges={"a": (27.0, 33.0), "b": (2.5, 3.5), "c": (25.0, 31.0)},
        default_ic_range=(-5.0, 5.0),
    ),
    # 24. RabinovichFabrikant
    "rabinovich_fabrikant": SystemConfig(
        name="RabinovichFabrikant",
        dim=3,
        default_params={"alpha": 0.14, "gamma": 0.1},
        param_ranges={"alpha": (0.12, 0.16), "gamma": (0.08, 0.12)},
        default_ic_range=(-0.5, 0.5),
    ),
    # 25. RikitakeDynamo
    "rikitake_dynamo": SystemConfig(
        name="RikitakeDynamo",
        dim=3,
        default_params={"a": 5.0, "mu": 2.0},
        param_ranges={"a": (4.5, 5.5), "mu": (1.8, 2.2)},
        default_ic_range=(-3.0, 3.0),
    ),
    # 26. Rossler
    "rossler": SystemConfig(
        name="Rossler",
        dim=3,
        default_params={"a": 0.2, "b": 0.2, "c": 5.7},
        param_ranges={"a": (0.15, 0.25), "b": (0.15, 0.25), "c": (5.0, 6.5)},
        default_ic_range=(-5.0, 5.0),
    ),
    # 27. Rucklidge
    "rucklidge": SystemConfig(
        name="Rucklidge",
        dim=3,
        default_params={"a": 2.0, "b": 6.7},
        param_ranges={"a": (1.8, 2.2), "b": (6.0, 7.4)},
        default_ic_range=(-2.0, 2.0),
    ),
    # 28. ShimizuMorioka
    "shimizu_morioka": SystemConfig(
        name="ShimizuMorioka",
        dim=3,
        default_params={"a": 0.75, "b": 0.45},
        param_ranges={"a": (0.7, 0.8), "b": (0.4, 0.5)},
        default_ic_range=(-1.0, 1.0),
    ),
    # 29. SprottJerk
    "sprott_jerk": SystemConfig(
        name="SprottJerk",
        dim=3,
        default_params={"a": 2.017},
        param_ranges={"a": (2.0, 2.05)},  # Tighter range for stability
        default_ic_range=(-0.3, 0.3),  # Smaller ICs for stability
    ),
    # 30. SprottLinzSprott (also known as SprottMore or Linz-Sprott)
    "sprott_linz": SystemConfig(
        name="SprottMore",
        dim=3,
        default_params={},  # Parameter-free system
        param_ranges={},
        default_ic_range=(-1.0, 1.0),
    ),
    # 31. Sprott M - one of Sprott's catalog of chaotic flows
    "sprott_m": SystemConfig(
        name="SprottM",
        dim=3,
        default_params={"a": 1.7, "b": 1.7},
        param_ranges={"a": (1.5, 1.9), "b": (1.5, 1.9)},
        default_ic_range=(-0.5, 0.5),
    ),
    # 32. StickSlipOscillator
    "stick_slip": SystemConfig(
        name="StickSlipOscillator",
        dim=3,
        default_params={
            "a": 1.0,
            "b": 0.5,
            "eps": 0.1,
            "mu_s": 0.5,
            "mu_d": 0.3,
            "v0": 1.0,
        },
        param_ranges={
            "a": (0.9, 1.1),
            "b": (0.4, 0.6),
            "eps": (0.08, 0.12),
            "mu_s": (0.45, 0.55),
            "mu_d": (0.25, 0.35),
            "v0": (0.9, 1.1),
        },
        default_ic_range=(-0.5, 0.5),
    ),
    # 32. SwingingAtwood
    "swinging_atwood": SystemConfig(
        name="SwingingAtwood",
        dim=4,  # 4D: (r, theta, pr, ptheta)
        default_params={"m1": 1.0, "m2": 4.5},
        param_ranges={"m1": (0.9, 1.1), "m2": (4.0, 5.0)},
        default_ic_range=(0.5, 2.0),
    ),
    # 33. Thomas
    "thomas": SystemConfig(
        name="Thomas",
        dim=3,
        default_params={"b": 0.208186},
        param_ranges={"b": (0.18, 0.24)},
        default_ic_range=(-2.0, 2.0),
    ),
    # 34. Windmi (atmosphere-ionosphere-magnetosphere model)
    "windmi": SystemConfig(
        name="Windmi",
        dim=3,
        default_params={"a": 0.7, "b": 2.5},
        param_ranges={"a": (0.6, 0.8), "b": (2.2, 2.8)},
        default_ic_range=(-1.0, 1.0),
    ),
    # ==== Additional 2D systems for completeness ====
    "vanderpol": SystemConfig(
        name="VanDerPol",
        dim=2,
        default_params={"mu": 1.0},
        param_ranges={"mu": (0.5, 5.0)},
        default_ic_range=(-2.0, 2.0),
    ),
    "selkov": SystemConfig(
        name="Selkov",
        dim=2,
        default_params={"a": 0.08, "b": 0.6},
        param_ranges={"a": (0.05, 0.15), "b": (0.4, 0.8)},
        default_ic_range=(0.1, 2.0),
    ),
    "duffing": SystemConfig(
        name="Duffing",
        dim=2,
        default_params={
            "alpha": 1.0,
            "beta": -1.0,
            "delta": 0.3,
            "gamma": 0.37,
            "omega": 1.2,
        },
        param_ranges={
            "alpha": (0.5, 1.5),
            "beta": (-1.5, -0.5),
            "delta": (0.1, 0.5),
            "gamma": (0.2, 0.5),
            "omega": (1.0, 1.4),
        },
        default_ic_range=(-2.0, 2.0),
    ),
}


# =============================================================================
# ODE implementations for all 34 training systems
# Equations from the dysts benchmark (Gilpin, 2021)
# =============================================================================


def aizawa(
    t: float,
    state: np.ndarray,
    a: float = 0.95,
    b: float = 0.7,
    c: float = 0.6,
    d: float = 3.5,
    e: float = 0.25,
    f: float = 0.1,
) -> np.ndarray:
    """Aizawa attractor - torus-like chaotic attractor."""
    x, y, z = state
    return np.array(
        [
            (z - b) * x - d * y,
            d * x + (z - b) * y,
            c + a * z - z**3 / 3 - (x**2 + y**2) * (1 + e * z) + f * z * x**3,
        ]
    )


def anishchenko_astakhov(
    t: float, state: np.ndarray, mu: float = 1.2, eta: float = 0.5
) -> np.ndarray:
    """Anishchenko-Astakhov oscillator - self-oscillating system."""
    x, y, z = state
    # Heaviside function approximation
    I_x = 1.0 if x > 0 else 0.0
    return np.array([mu * x + y - x * z, -x, -eta * z + eta * I_x * x**2])


def arneodo(
    t: float, state: np.ndarray, a: float = -5.5, b: float = 3.5, c: float = -1.0
) -> np.ndarray:
    """Arneodo system - third-order jerk system."""
    x, y, z = state
    return np.array([y, z, -a * x - b * y - z + c * x**3])


def belousov_zhabotinsky(
    t: float,
    state: np.ndarray,
    eps1: float = 0.001,
    eps2: float = 0.1,
    q: float = 0.01,
    f: float = 1.0,
) -> np.ndarray:
    """Belousov-Zhabotinsky reaction (Oregonator model)."""
    x, y, z = state
    # Ensure positive concentrations
    x = max(x, 1e-10)
    y = max(y, 1e-10)
    z = max(z, 1e-10)
    return np.array(
        [
            (q * y - x * y + x * (1 - x)) / eps1,
            (-q * y - x * y + f * z) / eps2,
            x - z,
        ]
    )


def chen(
    t: float, state: np.ndarray, a: float = 35.0, b: float = 3.0, c: float = 28.0
) -> np.ndarray:
    """Chen system - unified chaotic system."""
    x, y, z = state
    return np.array([a * (y - x), (c - a) * x - x * z + c * y, x * y - b * z])


def chua(
    t: float,
    state: np.ndarray,
    alpha: float = 15.6,
    beta: float = 28.0,
    m0: float = -1.143,
    m1: float = -0.714,
) -> np.ndarray:
    """Chua circuit - piecewise-linear chaotic electronic oscillator.

    The classic double-scroll attractor discovered by Leon Chua.
    Uses the piecewise-linear resistor characteristic:
    f(x) = m1*x + 0.5*(m0-m1)*(|x+1| - |x-1|)

    Standard parameters produce the famous double-scroll attractor.
    """
    x, y, z = state
    # Piecewise-linear function for Chua's diode
    fx = m1 * x + 0.5 * (m0 - m1) * (np.abs(x + 1) - np.abs(x - 1))
    return np.array([alpha * (y - x - fx), x - y + z, -beta * y])


def colpitts(
    t: float,
    state: np.ndarray,
    a: float = 4.0,
    b: float = 1.0,
    c: float = 0.07,
    d: float = 0.2,
) -> np.ndarray:
    """Colpitts oscillator - electronic circuit oscillator.

    Simplified parameters from dysts for better numerical stability.
    """
    x, y, z = state
    return np.array(
        [
            y,
            z,
            -a * (np.tanh(b * x) + c * y + d * z),
        ]
    )


def forced_brusselator(
    t: float,
    state: np.ndarray,
    a: float = 0.4,
    b: float = 1.2,
    f: float = 0.05,
    omega: float = 0.81,
) -> np.ndarray:
    """Forced Brusselator - chemical oscillator with forcing."""
    x, y, z = state
    return np.array(
        [
            a + x**2 * y - (b + 1) * x + f * np.cos(z),
            b * x - x**2 * y,
            omega,
        ]
    )


def forced_vanderpol(
    t: float,
    state: np.ndarray,
    mu: float = 8.53,
    a: float = 1.2,
    omega: float = 0.63,
) -> np.ndarray:
    """Forced Van der Pol oscillator - parametrically forced."""
    x, y, z = state
    return np.array(
        [
            y,
            mu * (1 - x**2) * y - x + a * np.sin(z),
            omega,
        ]
    )


def glycolytic_oscillation(
    t: float,
    state: np.ndarray,
    a: float = 100.0,
    b: float = 0.1,
    c: float = 4.0,
    d: float = 4.0,
    e: float = 0.02,
    f: float = 2.0,
    g: float = 3.5,
) -> np.ndarray:
    """Glycolytic oscillation - biochemical pathway model."""
    x, y, z = state
    # Ensure positive concentrations
    x = max(x, 1e-10)
    y = max(y, 1e-10)
    z = max(z, 1e-10)
    return np.array(
        [
            a * (b - x * y**2),
            x * y**2 - y - c * y * z / (d + y),
            e * (y - f * z),
        ]
    )


def hastings_powell(
    t: float,
    state: np.ndarray,
    a1: float = 5.0,
    a2: float = 0.1,
    b1: float = 3.0,
    b2: float = 2.0,
    d1: float = 0.4,
    d2: float = 0.01,
) -> np.ndarray:
    """Hastings-Powell food chain - three-species ecosystem."""
    x, y, z = state
    # Ensure positive populations
    x = max(x, 1e-10)
    y = max(y, 1e-10)
    z = max(z, 1e-10)
    return np.array(
        [
            x * (1 - x) - a1 * x * y / (1 + b1 * x),
            a1 * x * y / (1 + b1 * x) - a2 * y * z / (1 + b2 * y) - d1 * y,
            a2 * y * z / (1 + b2 * y) - d2 * z,
        ]
    )


def henon_heiles(t: float, state: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """Hénon-Heiles system - Hamiltonian chaos in galactic dynamics."""
    x, y, px, py = state
    return np.array(
        [
            px,
            py,
            -x - 2 * lam * x * y,
            -y - lam * (x**2 - y**2),
        ]
    )


def hopfield(
    t: float,
    state: np.ndarray,
    a: float = 1.0,
    b: float = 3.0,
    c: float = 1.0,
    d: float = 5.0,
    e: float = 0.16,
) -> np.ndarray:
    """Hopfield neural network - chaotic neural dynamics."""
    x, y, z = state
    return np.array(
        [
            -x + a * np.tanh(b * y),
            -y + a * np.tanh(b * z),
            -z + a * np.tanh(b * x) + d * np.sin(e * t)
            if t > 0
            else -z + a * np.tanh(b * x),
        ]
    )


def kawczynski_strizhak(
    t: float,
    state: np.ndarray,
    beta: float = 2.5,
    gamma: float = 0.05,
    kappa: float = 0.1,
    mu: float = 0.9,
) -> np.ndarray:
    """Kawczynski-Strizhak chemical oscillator.

    Correct formulation from dysts benchmark.
    """
    x, y, z = state
    return np.array(
        [
            gamma * y - gamma * x**3 + 3 * mu * gamma * x,
            -2 * mu * x - y - z + beta,
            kappa * x - kappa * z,
        ]
    )


def liu_chen(
    t: float,
    state: np.ndarray,
    a: float = 0.4,
    b: float = 12.0,
    c: float = 5.0,
    d: float = 1.0,
    e: float = 1.0,
    f: float = 4.0,
) -> np.ndarray:
    """Liu-Chen system - hyperchaotic variant."""
    x, y, z = state
    return np.array(
        [
            -a * x - y**2 + b * y * z,
            y - c * x * z + d * z,
            e * x * y - f * z,
        ]
    )


def lorenz(
    t: float,
    state: np.ndarray,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> np.ndarray:
    """Lorenz system - the canonical chaotic attractor."""
    x, y, z = state
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


def lorenz_bounded(
    t: float,
    state: np.ndarray,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> np.ndarray:
    """Lorenz system bounded - same as Lorenz but trajectory bounded."""
    x, y, z = state
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


def lorenz_stenflo(
    t: float,
    state: np.ndarray,
    a: float = 2.0,
    b: float = 0.7,
    c: float = 26.0,
    d: float = 1.5,
) -> np.ndarray:
    """Lorenz-Stenflo system - 4D extension of Lorenz."""
    x, y, z, w = state
    return np.array(
        [
            a * (y - x) + d * w,
            c * x - x * z - y,
            x * y - b * z,
            -x - a * w,
        ]
    )


def lu_chen_cheng(
    t: float, state: np.ndarray, a: float = 36.0, b: float = 3.0, c: float = 20.0
) -> np.ndarray:
    """Lu-Chen-Cheng system - variant of Chen system."""
    x, y, z = state
    return np.array([a * (y - x), -x * z + c * y, x * y - b * z])


def newton_leipnik(
    t: float, state: np.ndarray, a: float = 0.4, b: float = 0.175
) -> np.ndarray:
    """Newton-Leipnik system - two-scroll attractor."""
    x, y, z = state
    return np.array(
        [
            -a * x + y + 10 * y * z,
            -x - 0.4 * y + 5 * x * z,
            b * z - 5 * x * y,
        ]
    )


def nuclear_quadrupole(
    t: float, state: np.ndarray, a: float = 1.0, b: float = 0.55, c: float = 0.2
) -> np.ndarray:
    """Nuclear quadrupole resonance - spin dynamics."""
    x, y, z = state
    return np.array(
        [
            -a * y - b * z,
            a * x + c * y * z,
            b * x - c * y**2 + c * z**2,
        ]
    )


def piecewise_circuit(
    t: float, state: np.ndarray, a: float = 0.6, b: float = 1.0
) -> np.ndarray:
    """Piecewise linear circuit - electronic chaos."""
    x, y, z = state
    # Piecewise linear function
    if x >= 1:
        g = b
    elif x <= -1:
        g = -b
    else:
        g = b * x
    return np.array([y, z, -a * (z + y + x - g)])


def qi(
    t: float,
    state: np.ndarray,
    a: float = 45.0,
    b: float = 10.0,
    c: float = 1.0,
    d: float = 10.0,
) -> np.ndarray:
    """Qi system - four-wing chaotic attractor."""
    x, y, z = state
    return np.array(
        [
            a * (y - x) + y * z,
            c * x + d * y - x * z,
            -b * z + x * y,
        ]
    )


def qi_chen(
    t: float, state: np.ndarray, a: float = 30.0, b: float = 3.0, c: float = 28.0
) -> np.ndarray:
    """Qi-Chen system - similar to Chen with additional coupling.

    Using more stable parameters closer to Chen system.
    """
    x, y, z = state
    # Clamp to prevent blowup
    x = np.clip(x, -100, 100)
    y = np.clip(y, -100, 100)
    z = np.clip(z, -100, 100)
    return np.array([a * (y - x), c * y - x * z, x * y - b * z])


def rabinovich_fabrikant(
    t: float, state: np.ndarray, alpha: float = 0.14, gamma: float = 0.1
) -> np.ndarray:
    """Rabinovich-Fabrikant system - modulational instability."""
    x, y, z = state
    return np.array(
        [
            y * (z - 1 + x**2) + gamma * x,
            x * (3 * z + 1 - x**2) + gamma * y,
            -2 * z * (alpha + x * y),
        ]
    )


def rikitake_dynamo(
    t: float, state: np.ndarray, a: float = 5.0, mu: float = 2.0
) -> np.ndarray:
    """Rikitake dynamo - geomagnetic field reversals."""
    x, y, z = state
    return np.array([-mu * x + z * y, -mu * y + (z - a) * x, 1 - x * y])


def rossler(
    t: float, state: np.ndarray, a: float = 0.2, b: float = 0.2, c: float = 5.7
) -> np.ndarray:
    """Rössler system - simple chaotic flow."""
    x, y, z = state
    return np.array([-y - z, x + a * y, b + z * (x - c)])


def rucklidge(
    t: float, state: np.ndarray, a: float = 2.0, b: float = 6.7
) -> np.ndarray:
    """Rucklidge system - double convection model."""
    x, y, z = state
    return np.array([-a * x + b * y - y * z, x, -z + y**2])


def shimizu_morioka(
    t: float, state: np.ndarray, a: float = 0.75, b: float = 0.45
) -> np.ndarray:
    """Shimizu-Morioka system - Lorenz-like attractor."""
    x, y, z = state
    return np.array([y, x - a * y - x * z, -b * z + x**2])


def sprott_jerk(t: float, state: np.ndarray, a: float = 2.017) -> np.ndarray:
    """Sprott jerk system - simplest chaotic jerk equation.

    ẍ = y
    ÿ = z
    z̈ = -az + y² - x

    The attractor is bounded for a ≈ 2.017 but numerical integration can
    diverge for some initial conditions. State clamping prevents divergence
    during integration, and divergent trajectories are rejected at the
    trajectory generation level.
    """
    x, y, z = state
    # Clamp state to prevent divergence - attractor is bounded within ~10
    x = np.clip(x, -50, 50)
    y = np.clip(y, -50, 50)
    z = np.clip(z, -50, 50)
    return np.array([y, z, -a * z + y**2 - x])


def sprott_linz(t: float, state: np.ndarray) -> np.ndarray:
    """SprottMore system from dysts benchmark.

    Parameter-free jerk-like chaotic system.
    """
    x, y, z = state
    return np.array([y, -x - np.sign(z) * y, y**2 - np.exp(-(x**2))])


def sprott_m(
    t: float, state: np.ndarray, a: float = 1.7, b: float = 1.7
) -> np.ndarray:
    """Sprott M system - one of Sprott's catalog of simple chaotic flows.

    From J.C. Sprott's classification of 3D chaotic systems with quadratic
    nonlinearities. System M has the equations:
        dx/dt = -z
        dy/dt = -x^2 - y
        dz/dt = a + bx + y

    Produces a chaotic attractor for a = b = 1.7.
    """
    x, y, z = state
    return np.array([-z, -x**2 - y, a + b * x + y])


def stick_slip(
    t: float,
    state: np.ndarray,
    a: float = 1.0,
    b: float = 0.5,
    eps: float = 0.1,
    mu_s: float = 0.5,
    mu_d: float = 0.3,
    v0: float = 1.0,
) -> np.ndarray:
    """Stick-slip oscillator - friction-induced chaos."""
    x, y, z = state
    # Smooth friction approximation
    v_rel = y - v0
    friction = mu_d * np.tanh(v_rel / eps) + (mu_s - mu_d) * v_rel / (
        1 + np.abs(v_rel / eps)
    )
    return np.array([y, -a * x - b * y - friction + z, -z + 0.1 * (y - v0)])


def swinging_atwood(
    t: float, state: np.ndarray, m1: float = 1.0, m2: float = 4.5
) -> np.ndarray:
    """Swinging Atwood machine - Hamiltonian system."""
    r, theta, pr, ptheta = state
    # Avoid singularities
    r = max(r, 0.1)
    M = m1 + m2
    g = 9.81
    return np.array(
        [
            pr / M,
            ptheta / (m1 * r**2),
            ptheta**2 / (m1 * r**3) - m2 * g + m1 * g * np.cos(theta),
            -m1 * g * r * np.sin(theta),
        ]
    )


def thomas(t: float, state: np.ndarray, b: float = 0.208186) -> np.ndarray:
    """Thomas cyclically symmetric attractor."""
    x, y, z = state
    return np.array([np.sin(y) - b * x, np.sin(z) - b * y, np.sin(x) - b * z])


def windmi(t: float, state: np.ndarray, a: float = 0.7, b: float = 2.5) -> np.ndarray:
    """WINDMI model - magnetosphere-ionosphere coupling."""
    x, y, z = state
    return np.array([y, z, -a * z - y + b - np.exp(x)])


# ==== Additional 2D systems for completeness ====


def vanderpol(t: float, state: np.ndarray, mu: float = 1.0) -> np.ndarray:
    """Van der Pol oscillator."""
    x, y = state
    return np.array([y, mu * (1 - x**2) * y - x])


def selkov(t: float, state: np.ndarray, a: float = 0.08, b: float = 0.6) -> np.ndarray:
    """Selkov glycolytic oscillator."""
    x, y = state
    return np.array([-x + a * y + x**2 * y, b - a * y - x**2 * y])


def duffing(
    t: float,
    state: np.ndarray,
    alpha: float = 1.0,
    beta: float = -1.0,
    delta: float = 0.3,
    gamma: float = 0.37,
    omega: float = 1.2,
) -> np.ndarray:
    """Duffing oscillator (forced)."""
    x, y = state
    return np.array(
        [y, -delta * y - alpha * x - beta * x**3 + gamma * np.cos(omega * t)]
    )


# =============================================================================
# Map system names to functions
# =============================================================================

SYSTEM_FUNCTIONS = {
    # 34 training systems from the DynaMix paper
    "aizawa": aizawa,
    "anishchenko_astakhov": anishchenko_astakhov,
    "arneodo": arneodo,
    "belousov_zhabotinsky": belousov_zhabotinsky,
    "chen": chen,
    "chua": chua,
    "colpitts": colpitts,
    "forced_brusselator": forced_brusselator,
    "forced_vanderpol": forced_vanderpol,
    "glycolytic_oscillation": glycolytic_oscillation,
    "hastings_powell": hastings_powell,
    "henon_heiles": henon_heiles,
    "hopfield": hopfield,
    "kawczynski_strizhak": kawczynski_strizhak,
    "liu_chen": liu_chen,
    "lorenz": lorenz,
    "lorenz_bounded": lorenz_bounded,
    "lorenz_stenflo": lorenz_stenflo,
    "lu_chen_cheng": lu_chen_cheng,
    "newton_leipnik": newton_leipnik,
    "nuclear_quadrupole": nuclear_quadrupole,
    "piecewise_circuit": piecewise_circuit,
    "qi": qi,
    "qi_chen": qi_chen,
    "rabinovich_fabrikant": rabinovich_fabrikant,
    "rikitake_dynamo": rikitake_dynamo,
    "rossler": rossler,
    "rucklidge": rucklidge,
    "shimizu_morioka": shimizu_morioka,
    "sprott_jerk": sprott_jerk,
    "sprott_linz": sprott_linz,
    "sprott_m": sprott_m,
    "stick_slip": stick_slip,
    "swinging_atwood": swinging_atwood,
    "thomas": thomas,
    "windmi": windmi,
    # Additional 2D systems
    "vanderpol": vanderpol,
    "selkov": selkov,
    "duffing": duffing,
}


# =============================================================================
# NONSTATIONARY DYNAMICAL SYSTEMS
# These systems introduce various forms of nonstationarity to address the
# limitation identified in the DynaMix paper (Section 5): "Changes in
# statistical properties over time, tipping points, or widely differing
# time scales in the data, impose severe difficulties for zero-shot forecasting"
# =============================================================================


class NonstationaryWrapper:
    """Wrapper to add nonstationarity to any base dynamical system."""

    def __init__(
        self,
        base_func,
        nonstationarity_type: str,
        **kwargs,
    ):
        """
        Initialize the nonstationary wrapper.

        Args:
            base_func: Base ODE function to wrap
            nonstationarity_type: One of:
                - 'trend': Add polynomial trend to output
                - 'parameter_drift': Slowly vary parameters over time
                - 'regime_switch': Discrete parameter switches
                - 'amplitude_modulation': Time-varying amplitude
            **kwargs: Type-specific parameters
        """
        self.base_func = base_func
        self.nonstationarity_type = nonstationarity_type
        self.kwargs = kwargs

    def __call__(self, t: float, state: np.ndarray, **params) -> np.ndarray:
        """Apply the base system with nonstationarity."""
        return self.base_func(t, state, **params)


def add_trend_to_trajectory(
    trajectory: np.ndarray,
    t: np.ndarray,
    trend_coeffs: Optional[np.ndarray] = None,
    trend_type: str = "polynomial",
) -> np.ndarray:
    """
    Add a trend to an existing trajectory (post-processing).

    Args:
        trajectory: Trajectory array of shape (T, dim)
        t: Time array of shape (T,)
        trend_coeffs: Coefficients for the trend. If None, random coefficients are generated.
                     For polynomial: [a0, a1, a2] for a0 + a1*t + a2*t^2
        trend_type: Type of trend ('polynomial', 'exponential', 'sinusoidal')

    Returns:
        Trajectory with trend added
    """
    T, dim = trajectory.shape
    t_normalized = (t - t[0]) / (t[-1] - t[0])  # Normalize to [0, 1]

    if trend_type == "polynomial":
        if trend_coeffs is None:
            # Random polynomial trend: a0 + a1*t + a2*t^2
            # Scale coefficients to be proportional to trajectory std
            std = np.std(trajectory, axis=0)
            trend_coeffs = np.column_stack(
                [
                    np.random.uniform(-0.5, 0.5, dim) * std,  # constant offset
                    np.random.uniform(-2.0, 2.0, dim) * std,  # linear
                    np.random.uniform(-1.0, 1.0, dim) * std,  # quadratic
                ]
            )
        trend = (
            trend_coeffs[:, 0]
            + trend_coeffs[:, 1] * t_normalized[:, None]
            + trend_coeffs[:, 2] * (t_normalized[:, None] ** 2)
        )
    elif trend_type == "exponential":
        if trend_coeffs is None:
            std = np.std(trajectory, axis=0)
            trend_coeffs = np.column_stack(
                [
                    np.random.uniform(0.5, 1.5, dim) * std,  # amplitude
                    np.random.uniform(0.5, 2.0, dim),  # rate
                ]
            )
        trend = trend_coeffs[:, 0] * (
            np.exp(trend_coeffs[:, 1] * t_normalized[:, None]) - 1
        )
    elif trend_type == "sinusoidal":
        if trend_coeffs is None:
            std = np.std(trajectory, axis=0)
            trend_coeffs = np.column_stack(
                [
                    np.random.uniform(0.5, 2.0, dim) * std,  # amplitude
                    np.random.uniform(0.5, 3.0, dim),  # frequency
                ]
            )
        trend = trend_coeffs[:, 0] * np.sin(
            2 * np.pi * trend_coeffs[:, 1] * t_normalized[:, None]
        )
    else:
        raise ValueError(f"Unknown trend type: {trend_type}")

    return trajectory + trend


def apply_amplitude_modulation(
    trajectory: np.ndarray,
    t: np.ndarray,
    modulation_type: str = "linear",
    modulation_params: Optional[Dict] = None,
) -> np.ndarray:
    """
    Apply time-varying amplitude modulation to a trajectory.

    Args:
        trajectory: Trajectory array of shape (T, dim)
        t: Time array of shape (T,)
        modulation_type: Type of modulation ('linear', 'exponential', 'sinusoidal')
        modulation_params: Parameters for the modulation

    Returns:
        Amplitude-modulated trajectory
    """
    T, dim = trajectory.shape
    t_normalized = (t - t[0]) / (t[-1] - t[0])

    if modulation_params is None:
        modulation_params = {}

    if modulation_type == "linear":
        # Linear amplitude change from a_start to a_end
        a_start = modulation_params.get("a_start", 0.5)
        a_end = modulation_params.get("a_end", 2.0)
        amplitude = a_start + (a_end - a_start) * t_normalized
    elif modulation_type == "exponential":
        # Exponential growth/decay
        rate = modulation_params.get("rate", 1.0)
        amplitude = np.exp(rate * t_normalized)
    elif modulation_type == "sinusoidal":
        # Sinusoidal modulation
        freq = modulation_params.get("freq", 2.0)
        depth = modulation_params.get("depth", 0.5)
        amplitude = 1.0 + depth * np.sin(2 * np.pi * freq * t_normalized)
    else:
        raise ValueError(f"Unknown modulation type: {modulation_type}")

    # Center trajectory, apply modulation, then recenter
    mean = np.mean(trajectory, axis=0, keepdims=True)
    centered = trajectory - mean
    modulated = centered * amplitude[:, None]

    return modulated + mean


def create_parameter_drift_system(
    base_system: str,
    drift_param: str,
    drift_start: float,
    drift_end: float,
    t_total: float,
):
    """
    Create a system with slowly drifting parameters.

    Args:
        base_system: Name of the base system
        drift_param: Parameter name to drift
        drift_start: Starting value of the parameter
        drift_end: Ending value of the parameter
        t_total: Total integration time (for normalization)

    Returns:
        ODE function with time-varying parameter
    """
    base_func = SYSTEM_FUNCTIONS[base_system]

    def drifting_system(t: float, state: np.ndarray, **params) -> np.ndarray:
        # Interpolate the drifting parameter
        t_normalized = min(t / t_total, 1.0)
        current_value = drift_start + (drift_end - drift_start) * t_normalized

        # Update the parameter
        drifting_params = params.copy()
        drifting_params[drift_param] = current_value

        return base_func(t, state, **drifting_params)

    return drifting_system


def create_regime_switching_system(
    base_system: str,
    switch_param: str,
    regime_values: List[float],
    switch_times: List[float],
):
    """
    Create a system with discrete parameter regime switches.

    Args:
        base_system: Name of the base system
        switch_param: Parameter name that switches between regimes
        regime_values: List of parameter values for each regime
        switch_times: List of times at which to switch regimes

    Returns:
        ODE function with regime-switching behavior
    """
    base_func = SYSTEM_FUNCTIONS[base_system]

    def switching_system(t: float, state: np.ndarray, **params) -> np.ndarray:
        # Determine current regime
        regime_idx = 0
        for i, switch_time in enumerate(switch_times):
            if t >= switch_time:
                regime_idx = i + 1

        # Clamp to valid regime
        regime_idx = min(regime_idx, len(regime_values) - 1)

        # Update the parameter
        switching_params = params.copy()
        switching_params[switch_param] = regime_values[regime_idx]

        return base_func(t, state, **switching_params)

    return switching_system


def generate_nonstationary_trajectory(
    base_system: str,
    t_span: Tuple[float, float],
    nonstationarity_type: str,
    dt: float = 0.01,
    initial_condition: Optional[np.ndarray] = None,
    params: Optional[Dict[str, float]] = None,
    transient_time: float = 0.0,
    nonstationary_params: Optional[Dict] = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a nonstationary trajectory from a base dynamical system.

    Args:
        base_system: Name of the base dynamical system
        t_span: Time interval (t_start, t_end)
        nonstationarity_type: Type of nonstationarity:
            - 'trend_polynomial': Add polynomial trend
            - 'trend_exponential': Add exponential trend
            - 'trend_sinusoidal': Add sinusoidal trend
            - 'amplitude_linear': Linear amplitude modulation
            - 'amplitude_exponential': Exponential amplitude modulation
            - 'amplitude_sinusoidal': Sinusoidal amplitude modulation
            - 'parameter_drift': Slowly drifting parameter
            - 'regime_switch': Discrete regime switches
        dt: Time step for output
        initial_condition: Initial state vector
        params: System parameters
        transient_time: Time to discard for transients
        nonstationary_params: Parameters specific to the nonstationarity type
        **kwargs: Additional arguments passed to generate_trajectory

    Returns:
        t: Time array
        trajectory: Nonstationary trajectory array of shape (T, dim)
    """
    if nonstationary_params is None:
        nonstationary_params = {}

    # For parameter drift and regime switch, we need to modify the ODE
    if nonstationarity_type == "parameter_drift":
        drift_param = nonstationary_params.get("drift_param")
        if drift_param is None:
            # Pick a random parameter from the system
            config = SYSTEM_CONFIGS[base_system]
            if config.param_ranges:
                drift_param = list(config.param_ranges.keys())[0]
            else:
                raise ValueError(f"System {base_system} has no parameters to drift")

        drift_range = nonstationary_params.get("drift_range")
        if drift_range is None:
            config = SYSTEM_CONFIGS[base_system]
            if drift_param in config.param_ranges:
                low, high = config.param_ranges[drift_param]
                drift_range = (low, high)
            else:
                default_val = config.default_params.get(drift_param, 1.0)
                drift_range = (default_val * 0.8, default_val * 1.2)

        t_total = t_span[1] - t_span[0] + transient_time

        # Create drifting system
        drifting_func = create_parameter_drift_system(
            base_system, drift_param, drift_range[0], drift_range[1], t_total
        )

        # Get config for the base system
        config = SYSTEM_CONFIGS[base_system]

        # Use the drifting function directly with solve_ivp
        actual_t_span = (t_span[0] - transient_time, t_span[1])
        t_eval = np.arange(actual_t_span[0], actual_t_span[1], dt)

        if initial_condition is None:
            ic_range = config.default_ic_range
            ic = np.random.uniform(ic_range[0], ic_range[1], config.dim)
        else:
            ic = initial_condition

        if params is None:
            params = config.default_params.copy()

        sol = solve_ivp(
            lambda t, y: drifting_func(t, y, **params),
            actual_t_span,
            ic,
            method="RK45",
            t_eval=t_eval,
            rtol=1e-6,
            atol=1e-8,
            max_step=dt * 10,
        )

        if not sol.success or np.any(~np.isfinite(sol.y)):
            raise RuntimeError(f"Integration failed: {sol.message}")

        # Remove transient
        transient_steps = int(transient_time / dt)
        t = sol.t[transient_steps:] - transient_time
        trajectory = sol.y[:, transient_steps:].T

        return t, trajectory

    elif nonstationarity_type == "regime_switch":
        switch_param = nonstationary_params.get("switch_param")
        if switch_param is None:
            config = SYSTEM_CONFIGS[base_system]
            if config.param_ranges:
                switch_param = list(config.param_ranges.keys())[0]
            else:
                raise ValueError(
                    f"System {base_system} has no parameters for regime switching"
                )

        regime_values = nonstationary_params.get("regime_values")
        if regime_values is None:
            config = SYSTEM_CONFIGS[base_system]
            if switch_param in config.param_ranges:
                low, high = config.param_ranges[switch_param]
                # Create 2-3 regimes
                n_regimes = np.random.randint(2, 4)
                regime_values = np.linspace(low, high, n_regimes).tolist()
            else:
                default_val = config.default_params.get(switch_param, 1.0)
                regime_values = [default_val * 0.9, default_val * 1.1]

        switch_times = nonstationary_params.get("switch_times")
        if switch_times is None:
            t_duration = t_span[1] - t_span[0]
            n_switches = len(regime_values) - 1
            switch_times = [
                t_span[0] + (i + 1) * t_duration / (n_switches + 1)
                for i in range(n_switches)
            ]

        # Create switching system
        switching_func = create_regime_switching_system(
            base_system, switch_param, regime_values, switch_times
        )

        # Get config for the base system
        config = SYSTEM_CONFIGS[base_system]

        actual_t_span = (t_span[0] - transient_time, t_span[1])
        t_eval = np.arange(actual_t_span[0], actual_t_span[1], dt)

        if initial_condition is None:
            ic_range = config.default_ic_range
            ic = np.random.uniform(ic_range[0], ic_range[1], config.dim)
        else:
            ic = initial_condition

        if params is None:
            params = config.default_params.copy()

        sol = solve_ivp(
            lambda t, y: switching_func(t, y, **params),
            actual_t_span,
            ic,
            method="RK45",
            t_eval=t_eval,
            rtol=1e-6,
            atol=1e-8,
            max_step=dt * 10,
        )

        if not sol.success or np.any(~np.isfinite(sol.y)):
            raise RuntimeError(f"Integration failed: {sol.message}")

        transient_steps = int(transient_time / dt)
        t = sol.t[transient_steps:] - transient_time
        trajectory = sol.y[:, transient_steps:].T

        return t, trajectory

    else:
        # For trend and amplitude modulation, generate base trajectory first
        t, trajectory = generate_trajectory(
            system=base_system,
            t_span=t_span,
            dt=dt,
            initial_condition=initial_condition,
            params=params,
            transient_time=transient_time,
            **kwargs,
        )

        # Apply post-processing transformations
        if nonstationarity_type.startswith("trend_"):
            trend_type = nonstationarity_type.replace("trend_", "")
            trajectory = add_trend_to_trajectory(
                trajectory,
                t,
                trend_coeffs=nonstationary_params.get("trend_coeffs"),
                trend_type=trend_type,
            )
        elif nonstationarity_type.startswith("amplitude_"):
            modulation_type = nonstationarity_type.replace("amplitude_", "")
            trajectory = apply_amplitude_modulation(
                trajectory,
                t,
                modulation_type=modulation_type,
                modulation_params=nonstationary_params.get("modulation_params"),
            )
        else:
            raise ValueError(f"Unknown nonstationarity type: {nonstationarity_type}")

        return t, trajectory


def generate_nonstationary_training_data(
    base_systems: Union[str, List[str]],
    nonstationarity_types: Union[str, List[str]],
    n_trajectories_per_combination: int = 50,
    sequence_length: int = 550,
    context_length: int = 500,
    dt: float = 0.01,
    vary_params: bool = True,
    noise_level: float = 0.0,
    transient_time: float = 100.0,
    seed: Optional[int] = None,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate nonstationary training data for DynaMix.

    Creates data with various types of nonstationarity to help the model
    learn to handle nonstationary time series.

    Args:
        base_systems: Base system name(s) to use
        nonstationarity_types: Type(s) of nonstationarity to apply:
            - 'trend_polynomial', 'trend_exponential', 'trend_sinusoidal'
            - 'amplitude_linear', 'amplitude_exponential', 'amplitude_sinusoidal'
            - 'parameter_drift', 'regime_switch'
        n_trajectories_per_combination: Number of trajectories per system-nonstationarity pair
        sequence_length: Total sequence length
        context_length: Context length
        dt: Time step
        vary_params: Vary system parameters
        noise_level: Gaussian noise level
        transient_time: Transient time to discard
        seed: Random seed
        show_progress: Show progress bar

    Returns:
        data: Training data array
        context: Context array
        test: Test array
    """
    if seed is not None:
        np.random.seed(seed)

    if isinstance(base_systems, str):
        base_systems = [base_systems]
    if isinstance(nonstationarity_types, str):
        nonstationarity_types = [nonstationarity_types]

    all_trajectories = []
    t_end = (sequence_length + 10) * dt

    combinations = [
        (sys, ns_type) for sys in base_systems for ns_type in nonstationarity_types
    ]

    if show_progress:
        from tqdm import tqdm

        combinations = tqdm(combinations, desc="Generating nonstationary data")

    for base_system, ns_type in combinations:
        config = SYSTEM_CONFIGS[base_system]

        for _ in range(n_trajectories_per_combination):
            try:
                # Vary parameters if requested
                if vary_params and config.param_ranges:
                    traj_params = {}
                    for key, (low, high) in config.param_ranges.items():
                        traj_params[key] = np.random.uniform(low, high)
                else:
                    traj_params = None

                _, traj = generate_nonstationary_trajectory(
                    base_system=base_system,
                    t_span=(0, t_end),
                    nonstationarity_type=ns_type,
                    dt=dt,
                    params=traj_params,
                    transient_time=transient_time,
                )
                all_trajectories.append(traj[:sequence_length])
            except RuntimeError:
                # Try again with default params
                try:
                    _, traj = generate_nonstationary_trajectory(
                        base_system=base_system,
                        t_span=(0, t_end),
                        nonstationarity_type=ns_type,
                        dt=dt,
                        params=config.default_params,
                        transient_time=transient_time,
                    )
                    all_trajectories.append(traj[:sequence_length])
                except RuntimeError:
                    continue

    if len(all_trajectories) == 0:
        raise RuntimeError("All nonstationary trajectory generations failed!")

    # Stack trajectories: (T, S, N)
    trajectories = np.stack(all_trajectories, axis=1)

    # Standardize
    trajectories = standardize_trajectories(trajectories)

    # Add noise
    if noise_level > 0:
        std = np.std(trajectories, axis=0, keepdims=True)
        noise = np.random.randn(*trajectories.shape) * noise_level * std
        trajectories = trajectories + noise

    # Split into context and data
    delta_t = 50
    context = trajectories[:context_length].copy()
    data_start = context_length - delta_t
    data = trajectories[data_start:].copy()
    test = trajectories.copy()

    return data, context, test


# List of nonstationarity types for easy reference
NONSTATIONARITY_TYPES = [
    "trend_polynomial",
    "trend_exponential",
    "trend_sinusoidal",
    "amplitude_linear",
    "amplitude_exponential",
    "amplitude_sinusoidal",
    "parameter_drift",
    "regime_switch",
]

# Recommended base systems for nonstationary experiments (stable systems)
NONSTATIONARY_BASE_SYSTEMS = [
    "lorenz",
    "rossler",
    "chen",
    "thomas",
    "aizawa",
    "shimizu_morioka",
    "lu_chen_cheng",
    "rucklidge",
]


# Convenience list of all 34 3D training systems
TRAINING_SYSTEMS_3D = [
    "aizawa",
    "anishchenko_astakhov",
    "arneodo",
    "belousov_zhabotinsky",
    "chen",
    "chua",
    "colpitts",
    "forced_brusselator",
    "forced_vanderpol",
    "glycolytic_oscillation",
    "hastings_powell",
    "hopfield",
    "kawczynski_strizhak",
    "liu_chen",
    "lorenz",
    "lorenz_bounded",
    "lu_chen_cheng",
    "newton_leipnik",
    "nuclear_quadrupole",
    "piecewise_circuit",
    "qi",
    "qi_chen",
    "rabinovich_fabrikant",
    "rikitake_dynamo",
    "rossler",
    "rucklidge",
    "shimizu_morioka",
    "sprott_jerk",
    "sprott_linz",
    "sprott_m",
    "stick_slip",
    "thomas",
    "windmi",
]

# 4D systems (need special handling for 3D training)
TRAINING_SYSTEMS_4D = [
    "henon_heiles",
    "lorenz_stenflo",
    "swinging_atwood",
]


def generate_trajectory(
    system: str,
    t_span: Tuple[float, float],
    dt: float = 0.01,
    initial_condition: Optional[np.ndarray] = None,
    params: Optional[Dict[str, float]] = None,
    transient_time: float = 0.0,
    method: str = "RK45",
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_retries: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a trajectory from a dynamical system.

    Args:
        system: Name of the dynamical system (e.g., 'lorenz', 'rossler')
        t_span: Time interval (t_start, t_end)
        dt: Time step for output
        initial_condition: Initial state vector (if None, random IC is used)
        params: System parameters (if None, defaults are used)
        transient_time: Time to discard at the beginning to remove transients
        method: Integration method for solve_ivp
        rtol: Relative tolerance for solver
        atol: Absolute tolerance for solver
        max_retries: Maximum retries with different initial conditions

    Returns:
        t: Time array
        trajectory: State trajectory array of shape (T, dim)
    """
    if system not in SYSTEM_FUNCTIONS:
        raise ValueError(
            f"Unknown system: {system}. Available: {list(SYSTEM_FUNCTIONS.keys())}"
        )

    config = SYSTEM_CONFIGS[system]
    func = SYSTEM_FUNCTIONS[system]

    # Get parameters
    if params is None:
        params = config.default_params.copy()

    # Adjust time span to include transient
    actual_t_span = (t_span[0] - transient_time, t_span[1])

    # Create evaluation times
    t_eval = np.arange(actual_t_span[0], actual_t_span[1], dt)

    last_error = None
    # Try different methods - order depends on system characteristics
    # LSODA handles stiff systems, DOP853 is better for smooth non-stiff systems
    if system == "sprott_jerk":
        # sprott_jerk works better with explicit methods
        methods_to_try = ["DOP853", "RK45", "LSODA"]
    elif method == "LSODA":
        methods_to_try = ["LSODA", "DOP853"]
    else:
        methods_to_try = [method, "LSODA", "DOP853"]

    for attempt in range(max_retries):
        # Generate initial condition if not provided (or retry with new one)
        if initial_condition is None or attempt > 0:
            ic_range = config.default_ic_range
            ic = np.random.uniform(ic_range[0], ic_range[1], config.dim)
        else:
            ic = initial_condition

        # Try different integration methods
        for try_method in methods_to_try:
            try:
                # Integrate with max_step to prevent hanging on stiff problems
                sol = solve_ivp(
                    lambda t, y: func(t, y, **params),
                    actual_t_span,
                    ic,
                    method=try_method,
                    t_eval=t_eval,
                    rtol=rtol,
                    atol=atol,
                    max_step=dt * 10,  # Limit step size to prevent hanging
                )

                if sol.success:
                    # Check for NaN or Inf values
                    if np.any(~np.isfinite(sol.y)):
                        last_error = "Integration produced NaN/Inf values"
                        continue

                    # Check for divergence (trajectory values too large)
                    max_val = np.max(np.abs(sol.y))
                    if max_val > 1e6:
                        last_error = f"Trajectory diverged (max value: {max_val:.2e})"
                        continue

                    # Remove transient
                    transient_steps = int(transient_time / dt)
                    t = sol.t[transient_steps:] - transient_time
                    trajectory = sol.y[:, transient_steps:].T

                    return t, trajectory
                else:
                    last_error = sol.message
            except Exception as e:
                last_error = str(e)
                continue

    raise RuntimeError(f"Integration failed after {max_retries} attempts: {last_error}")


def generate_multi_trajectory(
    system: str,
    n_trajectories: int,
    t_span: Tuple[float, float],
    dt: float = 0.01,
    params: Optional[Dict[str, float]] = None,
    vary_params: bool = False,
    transient_time: float = 50.0,
    seed: Optional[int] = None,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Generate multiple trajectories from a dynamical system.

    Args:
        system: Name of the dynamical system
        n_trajectories: Number of trajectories to generate
        t_span: Time interval (t_start, t_end)
        dt: Time step
        params: Base system parameters
        vary_params: If True, randomly vary parameters within valid ranges
        transient_time: Time to discard for transients
        seed: Random seed
        show_progress: Show progress bar

    Returns:
        trajectories: Array of shape (T, n_trajectories, dim)
    """
    if seed is not None:
        np.random.seed(seed)

    config = SYSTEM_CONFIGS[system]
    trajectories = []

    iterator = range(n_trajectories)
    if show_progress:
        iterator = tqdm(iterator, desc=f"  {system}", leave=False)

    max_attempts_per_traj = 5
    failed_count = 0

    for _ in iterator:
        success = False
        for _ in range(max_attempts_per_traj):
            try:
                # Vary parameters if requested
                if vary_params and config.param_ranges:
                    traj_params = {}
                    for key, (low, high) in config.param_ranges.items():
                        traj_params[key] = np.random.uniform(low, high)
                else:
                    traj_params = params

                _, traj = generate_trajectory(
                    system=system,
                    t_span=t_span,
                    dt=dt,
                    params=traj_params,
                    transient_time=transient_time,
                )
                trajectories.append(traj)
                success = True
                break
            except RuntimeError:
                # Try again with different random params/IC
                continue

        if not success:
            failed_count += 1
            # Use default params as fallback
            try:
                _, traj = generate_trajectory(
                    system=system,
                    t_span=t_span,
                    dt=dt,
                    params=config.default_params,
                    transient_time=transient_time,
                )
                trajectories.append(traj)
            except RuntimeError as e:
                raise RuntimeError(
                    f"System {system} failed even with default params: {e}"
                )

    if failed_count > 0 and show_progress:
        print(
            f"    Warning: {failed_count}/{n_trajectories} trajectories needed fallback for {system}"
        )

    # Stack: (T, S, N)
    return np.stack(trajectories, axis=1)


def _init_worker(counter, lock):
    """Initialize worker process with shared counter."""
    global _shared_counter, _shared_lock
    _shared_counter = counter
    _shared_lock = lock


def _generate_system_data_worker(
    args: Tuple[str, int, int, float, bool, float, Optional[int]],
) -> Tuple[str, Optional[np.ndarray], int]:
    """
    Worker function to generate data for a single system.

    This function is designed to be called by multiprocessing.Pool.

    Args:
        args: Tuple of (system, n_trajectories, sequence_length, dt, vary_params,
              transient_time, seed)

    Returns:
        Tuple of (system_name, trajectories, failed_count) where:
        - system_name: Name of the system
        - trajectories: Array of shape (sequence_length, n_trajectories, dim) or None if all failed
        - failed_count: Number of trajectories that failed and were skipped
    """
    global _shared_counter, _shared_lock

    system, n_trajectories, sequence_length, dt, vary_params, transient_time, seed = (
        args
    )

    if seed is not None:
        np.random.seed(seed)

    config = SYSTEM_CONFIGS[system]
    t_end = (sequence_length + 10) * dt  # Extra buffer
    t_span = (0, t_end)

    trajectories = []
    max_attempts_per_traj = 10  # Increased from 5
    failed_count = 0

    for _ in range(n_trajectories):
        success = False
        for _ in range(max_attempts_per_traj):
            try:
                # Vary parameters if requested
                if vary_params and config.param_ranges:
                    traj_params = {}
                    for key, (low, high) in config.param_ranges.items():
                        traj_params[key] = np.random.uniform(low, high)
                else:
                    traj_params = None

                _, traj = generate_trajectory(
                    system=system,
                    t_span=t_span,
                    dt=dt,
                    params=traj_params,
                    transient_time=100.0,
                )
                trajectories.append(traj[:sequence_length])
                success = True
                break
            except RuntimeError:
                continue

        if not success:
            # Fallback to default params
            try:
                _, traj = generate_trajectory(
                    system=system,
                    t_span=t_span,
                    dt=dt,
                    params=config.default_params,
                    transient_time=100.0,
                )
                trajectories.append(traj[:sequence_length])
            except RuntimeError:
                # Skip this trajectory instead of crashing
                failed_count += 1

        # Update shared progress counter
        if _shared_counter is not None:
            with _shared_lock:
                _shared_counter.value += 1

    # Return None if all trajectories failed
    if len(trajectories) == 0:
        return (system, None, failed_count)

    # Stack: (T, S, N)
    return (system, np.stack(trajectories, axis=1), failed_count)


def generate_training_data(
    systems: Union[str, List[str]],
    n_trajectories_per_system: int = 100,
    sequence_length: int = 550,
    context_length: int = 500,
    dt: float = 0.01,
    vary_params: bool = True,
    noise_level: float = 0.0,
    transient_time: float = 100.0,
    seed: Optional[int] = None,
    show_progress: bool = True,
    n_workers: Optional[int] = None,
    save_intermediate: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate training data in the format expected by DynaMix.

    This creates three arrays:
    - data: Ground truth sequences of shape (T-T_C+Δt+1, S, N)
    - context: Context windows of shape (T_C, S, N)
    - test: Full test sequences of shape (T, S, N)

    Args:
        systems: System name or list of system names
        n_trajectories_per_system: Number of trajectories per system
        sequence_length: Total sequence length (T)
        context_length: Context length (T_C)
        dt: Time step
        vary_params: Vary system parameters for diversity
        noise_level: Gaussian noise level (fraction of std)
        transient_time: Transient time to discard
        seed: Random seed
        show_progress: Show progress bar
        n_workers: Number of parallel workers (default: None = sequential,
                   0 or negative = use all available CPUs)
        save_intermediate: Directory to save intermediate results per system.
                          If provided, data is saved after each system completes,
                          allowing recovery if generation fails partway through.

    Returns:
        data: Training data array
        context: Context array
        test: Test array
    """
    import os

    if seed is not None:
        np.random.seed(seed)

    if isinstance(systems, str):
        systems = [systems]

    all_trajectories = []
    failed_systems = []
    total_failed_trajectories = 0

    # Create intermediate save directory if specified
    if save_intermediate:
        os.makedirs(save_intermediate, exist_ok=True)
        intermediate_raw_dir = os.path.join(save_intermediate, "raw_systems")
        os.makedirs(intermediate_raw_dir, exist_ok=True)

    # Determine if we should use parallel processing
    use_parallel = n_workers is not None and len(systems) > 1

    if use_parallel:
        import time

        # Determine number of workers
        if n_workers <= 0:
            actual_workers = cpu_count()
        else:
            actual_workers = min(n_workers, len(systems), cpu_count())

        if show_progress:
            print(f"Using {actual_workers} parallel workers for {len(systems)} systems")

        # Create worker arguments with unique seeds for reproducibility
        worker_args = []
        for i, system in enumerate(systems):
            # Create unique seed for each system based on base seed
            system_seed = seed + i if seed is not None else None
            worker_args.append(
                (
                    system,
                    n_trajectories_per_system,
                    sequence_length,
                    dt,
                    vary_params,
                    transient_time,
                    system_seed,
                )
            )

        # Create shared counter for progress tracking
        manager = Manager()
        counter = manager.Value("i", 0)
        lock = manager.Lock()

        total_trajectories = len(systems) * n_trajectories_per_system

        # Use multiprocessing pool with shared counter
        with Pool(
            processes=actual_workers,
            initializer=_init_worker,
            initargs=(counter, lock),
        ) as pool:
            if show_progress:
                # Start async map
                async_result = pool.map_async(_generate_system_data_worker, worker_args)

                # Progress bar that polls the shared counter
                with tqdm(
                    total=total_trajectories, desc="Generating", unit="traj"
                ) as pbar:
                    last_count = 0
                    while not async_result.ready():
                        current_count = counter.value
                        if current_count > last_count:
                            pbar.update(current_count - last_count)
                            last_count = current_count
                        time.sleep(0.1)
                    # Final update
                    current_count = counter.value
                    if current_count > last_count:
                        pbar.update(current_count - last_count)

                results = async_result.get()
            else:
                results = pool.map(_generate_system_data_worker, worker_args)

        # Process results - handle failures gracefully
        for system_name, trajectories, failed_count in results:
            if trajectories is None:
                failed_systems.append(system_name)
                if show_progress:
                    print(
                        f"  Warning: System {system_name} failed completely ({failed_count} trajectories)"
                    )
            else:
                all_trajectories.append(trajectories)
                total_failed_trajectories += failed_count
                if failed_count > 0 and show_progress:
                    print(
                        f"  Warning: System {system_name} had {failed_count} failed trajectories"
                    )
                # Save intermediate result
                if save_intermediate:
                    np.save(
                        os.path.join(intermediate_raw_dir, f"{system_name}.npy"),
                        trajectories.astype(np.float32),
                    )

    else:
        # Sequential processing (original behavior)
        system_iterator = systems
        if show_progress:
            system_iterator = tqdm(systems, desc="Generating", unit="system")

        for system in system_iterator:
            t_end = (sequence_length + 10) * dt  # Extra buffer
            try:
                trajs = generate_multi_trajectory(
                    system=system,
                    n_trajectories=n_trajectories_per_system,
                    t_span=(0, t_end),
                    dt=dt,
                    vary_params=vary_params,
                    transient_time=transient_time,
                    seed=None,  # Don't reset seed for each system
                    show_progress=show_progress,
                )
                all_trajectories.append(trajs[:sequence_length])
                # Save intermediate result
                if save_intermediate:
                    np.save(
                        os.path.join(intermediate_raw_dir, f"{system}.npy"),
                        trajs[:sequence_length].astype(np.float32),
                    )
            except RuntimeError as e:
                failed_systems.append(system)
                if show_progress:
                    print(f"  Warning: System {system} failed: {e}")

    # Check if we have any successful data
    if len(all_trajectories) == 0:
        raise RuntimeError(f"All systems failed! Failed systems: {failed_systems}")

    # Report on failures
    if show_progress:
        if failed_systems:
            print(f"\nFailed systems ({len(failed_systems)}): {failed_systems}")
        if total_failed_trajectories > 0:
            print(f"Total failed trajectories: {total_failed_trajectories}")
        print(f"Successfully generated data from {len(all_trajectories)} systems")

    # Combine all systems: (T, S_total, N)
    trajectories = np.concatenate(all_trajectories, axis=1)

    # Standardize each trajectory
    trajectories = standardize_trajectories(trajectories)

    # Add noise if requested
    if noise_level > 0:
        std = np.std(trajectories, axis=0, keepdims=True)
        noise = np.random.randn(*trajectories.shape) * noise_level * std
        trajectories = trajectories + noise

    # Split into context and data
    # According to paper: Δt = 50 (overlap window)
    delta_t = 50

    # Context: first T_C timesteps
    context = trajectories[:context_length].copy()

    # Data: from context_length-delta_t onwards (for teacher forcing)
    # Shape should be (T-T_C+Δt+1, S, N)
    data_start = context_length - delta_t
    data = trajectories[data_start:].copy()

    # Test: full trajectories
    test = trajectories.copy()

    return data, context, test


def standardize_trajectories(trajectories: np.ndarray) -> np.ndarray:
    """
    Standardize trajectories to zero mean and unit variance per trajectory.

    Args:
        trajectories: Array of shape (T, S, N)

    Returns:
        Standardized trajectories
    """
    # Compute mean and std per trajectory (over time dimension)
    mean = np.mean(trajectories, axis=0, keepdims=True)
    std = np.std(trajectories, axis=0, keepdims=True)
    std[std < 1e-10] = 1.0  # Avoid division by zero

    return (trajectories - mean) / std


def save_training_data(
    save_dir: str,
    data: np.ndarray,
    context: np.ndarray,
    test: np.ndarray,
    prefix: str = "",
) -> Dict[str, str]:
    """
    Save training data to numpy files.

    Args:
        save_dir: Directory to save files
        data: Training data array
        context: Context array
        test: Test array
        prefix: Optional filename prefix

    Returns:
        Dictionary with paths to saved files
    """
    import os

    os.makedirs(save_dir, exist_ok=True)

    prefix = f"{prefix}_" if prefix else ""

    paths = {
        "data": os.path.join(save_dir, f"{prefix}data.npy"),
        "context": os.path.join(save_dir, f"{prefix}context.npy"),
        "test": os.path.join(save_dir, f"{prefix}test.npy"),
    }

    np.save(paths["data"], data.astype(np.float32))
    np.save(paths["context"], context.astype(np.float32))
    np.save(paths["test"], test.astype(np.float32))

    print("Saved training data:")
    print(f"  Data: {paths['data']} - shape {data.shape}")
    print(f"  Context: {paths['context']} - shape {context.shape}")
    print(f"  Test: {paths['test']} - shape {test.shape}")

    return paths


def list_available_systems() -> Dict[str, SystemConfig]:
    """Return dictionary of available systems and their configurations."""
    return SYSTEM_CONFIGS.copy()


def generate_single_system_file(
    system: str,
    n_timesteps: int = 12000,
    dt: float = 0.01,
    params: Optional[Dict[str, float]] = None,
    transient_time: float = 100.0,
    save_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a single trajectory file matching the test_data format.

    Args:
        system: Name of the dynamical system
        n_timesteps: Number of timesteps to generate
        dt: Time step
        params: System parameters (uses defaults if None)
        transient_time: Transient time to discard
        save_path: Optional path to save the trajectory
        seed: Random seed

    Returns:
        trajectory: Array of shape (n_timesteps, dim)
    """
    if seed is not None:
        np.random.seed(seed)

    t_end = (n_timesteps + 100) * dt

    _, trajectory = generate_trajectory(
        system=system,
        t_span=(0, t_end),
        dt=dt,
        params=params,
        transient_time=transient_time,
    )

    trajectory = trajectory[:n_timesteps]

    if save_path is not None:
        np.save(save_path, trajectory.astype(np.float64))
        print(f"Saved {system} trajectory to {save_path} - shape {trajectory.shape}")

    return trajectory


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate dynamical systems data for DynaMix training"
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=None,
        help="Systems to generate data from (default: TRAINING_SYSTEMS_3D)",
    )
    parser.add_argument(
        "--paper_systems",
        action="store_true",
        help="Use all 31 3D training systems from the DynaMix paper",
    )
    parser.add_argument(
        "--n_trajectories",
        type=int,
        default=100,
        help="Number of trajectories per system",
    )
    parser.add_argument(
        "--sequence_length", type=int, default=550, help="Sequence length (T)"
    )
    parser.add_argument(
        "--context_length", type=int, default=500, help="Context length (T_C)"
    )
    parser.add_argument("--dt", type=float, default=0.01, help="Time step")
    parser.add_argument(
        "--noise", type=float, default=0.0, help="Noise level (fraction of std)"
    )
    parser.add_argument(
        "--save_dir", type=str, default="training_data", help="Directory to save data"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: sequential, 0 = all CPUs)",
    )
    parser.add_argument(
        "--list_systems", action="store_true", help="List available systems and exit"
    )
    parser.add_argument(
        "--save_intermediate",
        action="store_true",
        help="Save intermediate results per system (enables recovery on failure)",
    )

    args = parser.parse_args()

    if args.list_systems:
        print("=" * 60)
        print("DynaMix Training Systems (34 total from the paper)")
        print("=" * 60)
        print("\n3D Training Systems (31 systems):")
        print("-" * 40)
        for name in TRAINING_SYSTEMS_3D:
            config = SYSTEM_CONFIGS[name]
            print(f"  {name}: {config.name}")
        print("\n4D Training Systems (3 systems - require projection to 3D):")
        print("-" * 40)
        for name in TRAINING_SYSTEMS_4D:
            config = SYSTEM_CONFIGS[name]
            print(f"  {name}: {config.name}")
        print("\n" + "=" * 60)
        print("Additional 2D Systems:")
        print("-" * 40)
        for name in ["vanderpol", "selkov", "duffing"]:
            config = SYSTEM_CONFIGS[name]
            print(f"  {name}: {config.name}")
        print("\nTo generate paper-matching training data, use:")
        print("  python -m dynamix.utilities.data_generation --paper_systems")
        exit(0)

    # Determine which systems to use
    if args.paper_systems:
        systems = TRAINING_SYSTEMS_3D
    elif args.systems is not None:
        systems = args.systems
    else:
        # Default to a few common systems
        systems = ["lorenz", "rossler", "chen"]

    print(f"Generating training data for {len(systems)} systems: {systems}")
    print(f"  Trajectories per system: {args.n_trajectories}")
    print(f"  Sequence length: {args.sequence_length}")
    print(f"  Context length: {args.context_length}")
    print(f"  Time step: {args.dt}")
    print(f"  Noise level: {args.noise}")
    if args.workers is not None:
        workers_str = "all CPUs" if args.workers <= 0 else str(args.workers)
        print(f"  Parallel workers: {workers_str}")
    if args.save_intermediate:
        print(f"  Intermediate saving: enabled (to {args.save_dir}/raw_systems/)")

    # Use save_dir for intermediate results if enabled
    intermediate_dir = args.save_dir if args.save_intermediate else None

    data, context, test = generate_training_data(
        systems=systems,
        n_trajectories_per_system=args.n_trajectories,
        sequence_length=args.sequence_length,
        context_length=args.context_length,
        dt=args.dt,
        noise_level=args.noise,
        seed=args.seed,
        n_workers=args.workers,
        save_intermediate=intermediate_dir,
    )

    save_training_data(args.save_dir, data, context, test)
    print("\nDone!")
