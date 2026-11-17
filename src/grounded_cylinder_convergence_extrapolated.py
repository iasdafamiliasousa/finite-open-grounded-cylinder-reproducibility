#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grounded_cylinder_convergence_extrapolated.py

Convergence analysis for the weighted Nystrom discretization of the
finite open grounded-cylinder problem.

The convergence error is estimated relative to an extrapolated
N -> infinity limit, rather than relative to the finest computed mesh.

The extrapolation model is

    F(N) = F_inf + c N^(-p),

where F_inf, c, and p are obtained by nonlinear least squares over the
asymptotic tail of the sequence.

Compatible with Google Colab, Jupyter, command-line execution, and import.
Outputs:
  - convergence_table_extrapolated.csv
  - convergence_table_extrapolated_SWP.tex
  - extrapolation_summary.txt
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.interpolate import BarycentricInterpolator
from scipy.optimize import curve_fit
from scipy.special import ellipk, ellipkm1


def parse_int_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("N_list cannot be empty.")
    if any(value < 20 for value in values):
        raise ValueError("All values in N_list must satisfy N >= 20.")
    return sorted(set(values))


def chebyshev_gauss_nodes(N: int) -> tuple[np.ndarray, float, np.ndarray]:
    j = np.arange(1, N + 1, dtype=float)
    theta = (2.0 * j - 1.0) * np.pi / (2.0 * N)
    x = np.cos(theta)
    w = np.pi / N
    return x, w, theta


def complete_elliptic_k(m: np.ndarray | float) -> np.ndarray:
    m_array = np.asarray(m, dtype=float)
    m_array = np.clip(m_array, 0.0, 1.0)

    p = 1.0 - m_array
    values = np.empty_like(m_array)

    near_one = p < 1.0e-10
    if np.any(near_one):
        values[near_one] = ellipkm1(p[near_one])
    if np.any(~near_one):
        values[~near_one] = ellipk(m_array[~near_one])

    bad = ~np.isfinite(values)
    if np.any(bad):
        p_bad = np.maximum(p[bad], 1.0e-300)
        values[bad] = np.log(4.0 / np.sqrt(p_bad))

    return values


def wall_kernel_from_squared_separation(s_squared: np.ndarray | float) -> np.ndarray:
    s_squared = np.asarray(s_squared, dtype=float)
    denominator = 4.0 + s_squared
    m = 4.0 / denominator
    return 4.0 * complete_elliptic_k(m) / np.sqrt(denominator)


def diagonal_cell_integrals(
    beta: float,
    x: np.ndarray,
    N: int,
    quadrature_order: int = 96,
) -> np.ndarray:
    theta = np.arccos(np.clip(np.asarray(x, dtype=float), -1.0, 1.0))
    cell_width = np.pi / N
    gl_x, gl_w = np.polynomial.legendre.leggauss(int(quadrature_order))
    diagonal = np.empty(N, dtype=float)

    for i, theta_i in enumerate(theta):
        lower = max(0.0, theta_i - 0.5 * cell_width)
        upper = min(np.pi, theta_i + 0.5 * cell_width)

        midpoint = 0.5 * (lower + upper)
        half_width = 0.5 * (upper - lower)

        theta_q = midpoint + half_width * gl_x
        weights_q = half_width * gl_w

        separation = beta * (math.cos(theta_i) - np.cos(theta_q))
        diagonal[i] = np.sum(
            weights_q * wall_kernel_from_squared_separation(separation**2)
        )

    return diagonal


@dataclass
class NumericalSolution:
    N: int
    beta: float
    u0: float
    x: np.ndarray
    theta: np.ndarray
    p: np.ndarray
    Qind_over_q: float
    collocation_residual: float


@dataclass
class ExtrapolationResult:
    F_inf: float
    c: float
    p: float
    sigma_F_inf: float
    fit_rms: float
    N_min_fit: int
    N_points_fit: int


def solve_finite_cylinder(
    L_over_a: float,
    z0_over_a: float,
    N: int,
    diagonal_quadrature_order: int = 96,
) -> NumericalSolution:
    L_over_a = float(L_over_a)
    z0_over_a = float(z0_over_a)
    N = int(N)
    beta = 0.5 * L_over_a

    if L_over_a <= 0.0:
        raise ValueError("L_over_a must be positive.")
    if N < 20:
        raise ValueError("N must satisfy N >= 20.")
    if not (-beta < z0_over_a < beta):
        raise ValueError(
            f"z0/a={z0_over_a:g} must satisfy {-beta:g} < z0/a < {beta:g}."
        )

    x, weight, theta = chebyshev_gauss_nodes(N)
    u = beta * x

    separation = u[:, None] - u[None, :]
    matrix = beta * weight * wall_kernel_from_squared_separation(separation**2)

    diagonal = diagonal_cell_integrals(
        beta=beta,
        x=x,
        N=N,
        quadrature_order=diagonal_quadrature_order,
    )
    np.fill_diagonal(matrix, beta * diagonal)

    rhs = -1.0 / np.sqrt(1.0 + (u - z0_over_a) ** 2)

    try:
        p = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        p, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)

    collocation_residual = float(np.max(np.abs(matrix @ p - rhs)))
    Qind_over_q = float(2.0 * np.pi * beta * weight * np.sum(p))

    return NumericalSolution(
        N=N,
        beta=beta,
        u0=z0_over_a,
        x=x,
        theta=theta,
        p=p,
        Qind_over_q=Qind_over_q,
        collocation_residual=collocation_residual,
    )


def regular_amplitude_interpolator(
    solution: NumericalSolution,
) -> BarycentricInterpolator:
    order = np.argsort(solution.x)
    return BarycentricInterpolator(solution.x[order], solution.p[order])


def density_at_u(solution: NumericalSolution, u_eval: float) -> float:
    x_eval = float(u_eval) / solution.beta
    if not (-1.0 < x_eval < 1.0):
        raise ValueError("The density evaluation point must be inside the cylinder.")

    interpolator = regular_amplitude_interpolator(solution)
    p_eval = float(interpolator(x_eval))
    return p_eval / math.sqrt(1.0 - x_eval**2)


def edge_amplitudes(solution: NumericalSolution) -> tuple[float, float]:
    interpolator = regular_amplitude_interpolator(solution)
    p_left = float(interpolator(-1.0))
    p_right = float(interpolator(1.0))
    factor = math.sqrt(solution.beta / 2.0)
    return factor * p_left, factor * p_right


def gauss_integral_on_interval(function, lower: float, upper: float, order: int) -> float:
    if upper <= lower:
        return 0.0

    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    midpoint = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)
    points = midpoint + half_width * nodes
    return float(half_width * np.sum(weights * function(points)))


def off_grid_boundary_residual(
    solution: NumericalSolution,
    number_of_test_points: int = 17,
    validation_quadrature_order: int = 100,
) -> float:
    interpolator = regular_amplitude_interpolator(solution)
    beta = solution.beta
    u0 = solution.u0

    test_theta = (
        np.arange(1, int(number_of_test_points) + 1, dtype=float)
        * np.pi
        / (int(number_of_test_points) + 1)
    )

    maximum_residual = 0.0

    for theta_i in test_theta:
        x_i = math.cos(theta_i)
        u_i = beta * x_i

        def integrand(theta_q: np.ndarray) -> np.ndarray:
            x_q = np.cos(theta_q)
            p_q = interpolator(x_q)
            separation = beta * (x_i - x_q)
            return p_q * wall_kernel_from_squared_separation(separation**2)

        integral = beta * (
            gauss_integral_on_interval(
                integrand, 0.0, theta_i, validation_quadrature_order
            )
            + gauss_integral_on_interval(
                integrand, theta_i, np.pi, validation_quadrature_order
            )
        )

        rhs = -1.0 / math.sqrt(1.0 + (u_i - u0) ** 2)
        maximum_residual = max(maximum_residual, abs(integral - rhs))

    return float(maximum_residual)


def convergence_model(N: np.ndarray, F_inf: float, c: float, p: float) -> np.ndarray:
    return F_inf + c * N ** (-p)


def extrapolate_to_infinity(
    N_values: Iterable[int],
    F_values: Iterable[float],
    tail_points: int = 5,
    p_bounds: tuple[float, float] = (0.1, 8.0),
) -> ExtrapolationResult:
    N = np.asarray(list(N_values), dtype=float)
    F = np.asarray(list(F_values), dtype=float)

    if N.size != F.size:
        raise ValueError("N_values and F_values must have the same length.")
    if N.size < 4:
        raise ValueError("At least four discretizations are required for extrapolation.")

    tail_points = int(tail_points)
    if tail_points < 4:
        raise ValueError("tail_points must be at least 4.")
    tail_points = min(tail_points, N.size)

    N_fit = N[-tail_points:]
    F_fit = F[-tail_points:]

    # Initial guess from the last three points.
    F_inf_0 = F_fit[-1]
    c_0 = (F_fit[0] - F_fit[-1]) * N_fit[0]
    p_0 = 1.0

    popt, pcov = curve_fit(
        convergence_model,
        N_fit,
        F_fit,
        p0=(F_inf_0, c_0, p_0),
        bounds=([-np.inf, -np.inf, p_bounds[0]],
                [ np.inf,  np.inf, p_bounds[1]]),
        maxfev=50000,
    )

    F_inf, c, p = [float(value) for value in popt]
    fitted = convergence_model(N_fit, *popt)
    fit_rms = float(np.sqrt(np.mean((F_fit - fitted) ** 2)))

    if pcov.shape == (3, 3) and np.isfinite(pcov[0, 0]) and pcov[0, 0] >= 0.0:
        sigma_F_inf = float(np.sqrt(pcov[0, 0]))
    else:
        sigma_F_inf = float("nan")

    return ExtrapolationResult(
        F_inf=F_inf,
        c=c,
        p=p,
        sigma_F_inf=sigma_F_inf,
        fit_rms=fit_rms,
        N_min_fit=int(N_fit[0]),
        N_points_fit=int(N_fit.size),
    )


def relative_error_to_extrapolated(value: float, extrapolated: float) -> float:
    if extrapolated == 0.0:
        return abs(value - extrapolated)
    return abs(value - extrapolated) / abs(extrapolated)


def scientific_latex(value: float, digits: int = 2) -> str:
    if not np.isfinite(value):
        return "---"
    if value == 0.0:
        return "$0$"

    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10.0**exponent)

    if -2 <= exponent <= 2:
        return f"${value:.{digits + 2}f}$"

    return f"${mantissa:.{digits}f}\\times 10^{{{exponent}}}$"


def write_csv(rows: list[dict[str, float]], path: str) -> None:
    fieldnames = [
        "N",
        "Qind_over_q",
        "rel_error_Q_extrapolated",
        "sigma_at_source",
        "rel_error_sigma_extrapolated",
        "C_left",
        "rel_error_C_left_extrapolated",
        "C_right",
        "rel_error_C_right_extrapolated",
        "validation_residual_inf",
        "collocation_residual_inf",
    ]

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def write_swp_latex_table(
    rows: list[dict[str, float]],
    path: str,
    L_over_a: float,
    z0_over_a: float,
    Q_fit: ExtrapolationResult,
    sigma_fit: ExtrapolationResult,
) -> None:
    lines = [
        r"\documentclass[12pt]{article}",
        r"",
        r"\usepackage{amsmath}",
        r"",
        r"\begin{document}",
        r"",
        r"\begin{table}[ht]",
        r"\centering",
        (
            r"\caption{Convergence of the weighted Nystr\"om discretization "
            rf"for $L/a={L_over_a:g}$ and $z_0/a={z0_over_a:g}$. "
            r"The errors are measured relative to extrapolated "
            r"$N\rightarrow\infty$ limits obtained from "
            r"$F(N)=F_{\infty}+cN^{-p}$.}"
        ),
        r"\label{tab:convergence}",
        r"",
        r"\begin{tabular}{rcccccc}",
        r"\hline",
        r"\hline",
        (
            r"$N$ & $Q_{\mathrm{ind}}/q$ & $\epsilon_Q$ & "
            r"$\sigma_L(u_0)$ & $\epsilon_{\sigma}$ & "
            r"$C_{-}$ & $R_{\infty}$ \\"
        ),
        r"\hline",
    ]

    for row in rows:
        lines.append(
            f"{int(row['N'])} & "
            f"${row['Qind_over_q']:.10f}$ & "
            f"{scientific_latex(row['rel_error_Q_extrapolated'], 2)} & "
            f"${row['sigma_at_source']:.10f}$ & "
            f"{scientific_latex(row['rel_error_sigma_extrapolated'], 2)} & "
            f"{scientific_latex(row['C_left'], 5)} & "
            f"{scientific_latex(row['validation_residual_inf'], 2)} \\\\"
        )

    lines.extend(
        [
            r"\hline",
            r"\hline",
            r"\end{tabular}",
            r"",
            (
                r"\begin{flushleft}"
                rf"$Q_{{\mathrm{{ind}}}}/q\rightarrow {Q_fit.F_inf:.12f}$, "
                rf"$p_Q={Q_fit.p:.4f}$; "
                rf"$\sigma_L(u_0)\rightarrow {sigma_fit.F_inf:.12f}$, "
                rf"$p_\sigma={sigma_fit.p:.4f}$."
                r"\end{flushleft}"
            ),
            r"\end{table}",
            r"",
            r"\end{document}",
            r"",
        ]
    )

    with open(path, "w", encoding="ascii") as file:
        file.write("\n".join(lines))


def write_summary(
    path: str,
    fits: dict[str, ExtrapolationResult],
) -> None:
    with open(path, "w", encoding="utf-8") as file:
        for name, fit in fits.items():
            file.write(f"{name}\n")
            file.write(f"  F_inf          = {fit.F_inf:.16e}\n")
            file.write(f"  c              = {fit.c:.16e}\n")
            file.write(f"  p              = {fit.p:.10f}\n")
            file.write(f"  sigma_F_inf    = {fit.sigma_F_inf:.6e}\n")
            file.write(f"  fit_rms        = {fit.fit_rms:.6e}\n")
            file.write(f"  N_min_fit      = {fit.N_min_fit}\n")
            file.write(f"  N_points_fit   = {fit.N_points_fit}\n\n")


def run_convergence_analysis(
    L_over_a: float = 6.0,
    z0_over_a: float = 0.0,
    N_values: Iterable[int] = (60, 80, 120, 180, 260, 360, 500, 700),
    output_directory: str = "convergence_out",
    diagonal_quadrature_order: int = 96,
    validation_test_points: int = 17,
    validation_quadrature_order: int = 100,
    extrapolation_tail_points: int = 5,
) -> tuple[list[dict[str, float]], dict[str, ExtrapolationResult], str, str]:
    os.makedirs(output_directory, exist_ok=True)

    N_values = sorted(set(int(N) for N in N_values))
    if len(N_values) < 4:
        raise ValueError("Use at least four N values.")

    rows: list[dict[str, float]] = []

    print("\nConvergence analysis with N -> infinity extrapolation")
    print(f"L/a = {L_over_a:g}")
    print(f"z0/a = {z0_over_a:g}")
    print(f"N values = {N_values}")
    print(f"tail points used in fit = {extrapolation_tail_points}\n")

    for N in N_values:
        solution = solve_finite_cylinder(
            L_over_a=L_over_a,
            z0_over_a=z0_over_a,
            N=N,
            diagonal_quadrature_order=diagonal_quadrature_order,
        )

        sigma_at_source = density_at_u(solution, z0_over_a)
        C_left, C_right = edge_amplitudes(solution)

        validation_residual = off_grid_boundary_residual(
            solution,
            number_of_test_points=validation_test_points,
            validation_quadrature_order=validation_quadrature_order,
        )

        rows.append(
            {
                "N": float(N),
                "Qind_over_q": solution.Qind_over_q,
                "sigma_at_source": sigma_at_source,
                "C_left": C_left,
                "C_right": C_right,
                "validation_residual_inf": validation_residual,
                "collocation_residual_inf": solution.collocation_residual,
            }
        )

        print(
            f"N={N:5d}  "
            f"Q_ind/q={solution.Qind_over_q:+.12e}  "
            f"sigma_L(u0)={sigma_at_source:+.12e}  "
            f"C_-={C_left:+.6e}  "
            f"R_inf={validation_residual:.3e}"
        )

    N_array = np.array([row["N"] for row in rows], dtype=float)

    fits = {
        "Qind_over_q": extrapolate_to_infinity(
            N_array,
            [row["Qind_over_q"] for row in rows],
            tail_points=extrapolation_tail_points,
        ),
        "sigma_at_source": extrapolate_to_infinity(
            N_array,
            [row["sigma_at_source"] for row in rows],
            tail_points=extrapolation_tail_points,
        ),
        "C_left": extrapolate_to_infinity(
            N_array,
            [row["C_left"] for row in rows],
            tail_points=extrapolation_tail_points,
        ),
        "C_right": extrapolate_to_infinity(
            N_array,
            [row["C_right"] for row in rows],
            tail_points=extrapolation_tail_points,
        ),
    }

    for row in rows:
        row["rel_error_Q_extrapolated"] = relative_error_to_extrapolated(
            row["Qind_over_q"], fits["Qind_over_q"].F_inf
        )
        row["rel_error_sigma_extrapolated"] = relative_error_to_extrapolated(
            row["sigma_at_source"], fits["sigma_at_source"].F_inf
        )
        row["rel_error_C_left_extrapolated"] = relative_error_to_extrapolated(
            row["C_left"], fits["C_left"].F_inf
        )
        row["rel_error_C_right_extrapolated"] = relative_error_to_extrapolated(
            row["C_right"], fits["C_right"].F_inf
        )

    csv_path = os.path.join(
        output_directory, "convergence_table_extrapolated.csv"
    )
    latex_path = os.path.join(
        output_directory, "convergence_table_extrapolated_SWP.tex"
    )
    summary_path = os.path.join(
        output_directory, "extrapolation_summary.txt"
    )

    write_csv(rows, csv_path)
    write_swp_latex_table(
        rows=rows,
        path=latex_path,
        L_over_a=L_over_a,
        z0_over_a=z0_over_a,
        Q_fit=fits["Qind_over_q"],
        sigma_fit=fits["sigma_at_source"],
    )
    write_summary(summary_path, fits)

    print("\nExtrapolated limits")
    for name, fit in fits.items():
        print(
            f"{name:20s}: F_inf={fit.F_inf:+.12e}, "
            f"p={fit.p:.5f}, fit_rms={fit.fit_rms:.3e}"
        )

    print(f"\nCSV table: {csv_path}")
    print(f"SWP LaTeX: {latex_path}")
    print(f"Summary:   {summary_path}")

    return rows, fits, csv_path, latex_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convergence analysis with extrapolation to N -> infinity "
            "for the finite grounded cylinder."
        )
    )
    parser.add_argument("--L_over_a", type=float, default=6.0)
    parser.add_argument("--z0_over_a", type=float, default=0.0)
    parser.add_argument(
        "--N_list",
        type=str,
        default="60,80,120,180,260,360,500,700",
    )
    parser.add_argument("--outdir", type=str, default="convergence_out")
    parser.add_argument("--diag_quad_order", type=int, default=96)
    parser.add_argument("--validation_test_points", type=int, default=17)
    parser.add_argument("--validation_quad_order", type=int, default=100)
    parser.add_argument("--fit_tail_points", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_argument_parser()
    args, unknown = parser.parse_known_args(args=argv)

    if unknown:
        print("Ignoring notebook arguments:", " ".join(unknown))

    run_convergence_analysis(
        L_over_a=args.L_over_a,
        z0_over_a=args.z0_over_a,
        N_values=parse_int_list(args.N_list),
        output_directory=args.outdir,
        diagonal_quadrature_order=args.diag_quad_order,
        validation_test_points=args.validation_test_points,
        validation_quadrature_order=args.validation_quad_order,
        extrapolation_tail_points=args.fit_tail_points,
    )


if __name__ == "__main__":
    main()