"""
montecarlo.py -- Monte Carlo impact dispersion and IIP analysis for traj3dof.

1. Fly the nominal gravity turn and extract its pitch program.
2. Fly N dispersed cases that track that pitch program open-loop.
3. Fit impact-point covariance and draw 1/2/3-sigma ellipses.
4. Trace the nominal IIP during powered flight against a notional destruct line.
"""

import argparse
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from traj3dof import Vehicle, Guidance, Wind, simulate, pitch_program, iip, RE

# 1-sigma dispersions (Gaussian). Assumed values -- documented in README.
SIGMA = {
    "thrust": 0.01,       # fraction
    "isp": 0.005,         # fraction
    "m_dry": 0.02,        # fraction of dry mass
    "cd": 0.05,           # fraction
    "yaw_err_deg": 0.1,   # deg
    "wind_dr": 7.0,       # m/s
    "wind_cr": 7.0,       # m/s
}


def disperse(rng, veh0, gd0):
    m_dry0 = veh0.m0 - veh0.m_prop
    m_dry = m_dry0 * (1 + SIGMA["m_dry"] * rng.standard_normal())
    veh = replace(veh0,
                  m0=m_dry + veh0.m_prop,
                  thrust=veh0.thrust * (1 + SIGMA["thrust"] * rng.standard_normal()),
                  isp=veh0.isp * (1 + SIGMA["isp"] * rng.standard_normal()),
                  cd=veh0.cd * (1 + SIGMA["cd"] * rng.standard_normal()))
    gd = replace(gd0, yaw_err_deg=SIGMA["yaw_err_deg"] * rng.standard_normal())
    wind = Wind(SIGMA["wind_dr"] * rng.standard_normal(),
                SIGMA["wind_cr"] * rng.standard_normal())
    return veh, gd, wind


def sigma_ellipse(mean, cov, k, n=200):
    """Points on the k-sigma ellipse of a 2D Gaussian."""
    vals, vecs = np.linalg.eigh(cov)
    ang = np.linspace(0, 2 * np.pi, n)
    circle = np.vstack([np.cos(ang), np.sin(ang)])
    return (mean[:, None] + vecs @ (k * np.sqrt(vals)[:, None] * circle))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=1000, help="number of Monte Carlo cases")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--destruct-km", type=float, default=320.0,
                    help="notional downrange destruct line for IIP [km]")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # ---- Nominal ----
    veh0 = Vehicle()
    nom = simulate(veh0, Guidance())
    gd0 = Guidance(pitch_table=pitch_program(nom))
    nom = simulate(veh0, gd0)
    print(f"Nominal: apogee {nom['apogee_alt']/1e3:.1f} km, "
          f"impact {nom['impact_dr']/1e3:.1f} km downrange")

    # ---- Monte Carlo ----
    impacts = np.zeros((args.n, 2))
    apogees = np.zeros(args.n)
    for i in range(args.n):
        veh, gd, wind = disperse(rng, veh0, gd0)
        r = simulate(veh, gd, wind)
        impacts[i] = r["impact_dr"], r["impact_cr"]
        apogees[i] = r["apogee_alt"]
        if (i + 1) % max(1, args.n // 10) == 0:
            print(f"  {i + 1}/{args.n} cases")

    imp_km = impacts / 1e3
    mean = imp_km.mean(axis=0)
    cov = np.cov(imp_km.T)
    sd = np.sqrt(np.diag(cov))
    print(f"\nImpact mean:  downrange {mean[0]:.2f} km, crossrange {mean[1]:.2f} km")
    print(f"Impact 1-sig: downrange {sd[0]:.2f} km, crossrange {sd[1]:.2f} km")
    print(f"Apogee:       {apogees.mean()/1e3:.1f} +/- {apogees.std()/1e3:.1f} km (1-sig)")

    # Containment check: fraction inside k-sigma ellipse vs theory 1 - exp(-k^2/2)
    diff = imp_km - mean
    d2 = np.einsum("ij,jk,ik->i", diff, np.linalg.inv(cov), diff)   # squared Mahalanobis
    for k in (1, 2, 3):
        print(f"  inside {k}-sigma ellipse: {np.mean(d2 <= k**2):6.1%}  "
              f"(2D Gaussian theory {1 - np.exp(-k**2 / 2):6.1%})")

    # Convergence: running estimate of downrange sigma vs N
    ns = np.unique(np.logspace(1, np.log10(args.n), 40).astype(int))
    run_sd = [imp_km[:k, 0].std(ddof=1) for k in ns]

    # ---- Nominal IIP trace ----
    t, S = nom["t"], nom["state"]
    powered = (t > 1.0) & (t <= nom["t_burnout"])
    iip_dr = np.array([iip(S[0:3, j], S[3:6, j], 90.0)[0] for j in np.where(powered)[0]]) / 1e3
    t_iip = t[powered]
    crossing = t_iip[iip_dr >= args.destruct_km]
    if crossing.size:
        print(f"\nNominal IIP crosses {args.destruct_km:.0f} km line at t = {crossing[0]:.1f} s")
    else:
        print(f"\nNominal IIP never crosses the {args.destruct_km:.0f} km line "
              f"(max {np.nanmax(iip_dr):.1f} km at burnout)")

    # Dispersed cases whose impact (= final IIP, ignoring reentry drag) exceeds the line
    n_over = int(np.sum(imp_km[:, 0] > args.destruct_km))
    p_hat = n_over / args.n
    se = np.sqrt(max(p_hat, 1 / args.n) * (1 - p_hat) / args.n)
    print(f"Dispersed cases beyond destruct line: {n_over}/{args.n} "
          f"= {p_hat:.2%} (+/- {se:.2%} std. error)")

    # ---- Plots ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))

    ax[0].scatter(imp_km[:, 0], imp_km[:, 1], s=4, alpha=0.4, label="MC impacts")
    for k, ls in zip((1, 2, 3), ("-", "--", ":")):
        e = sigma_ellipse(mean, cov, k)
        ax[0].plot(e[0], e[1], "k", ls=ls, label=f"{k}$\\sigma$")
    ax[0].plot(nom["impact_dr"] / 1e3, nom["impact_cr"] / 1e3, "r*", ms=12, label="nominal")
    ax[0].axvline(args.destruct_km, color="r", ls="--", lw=1, label="destruct line")
    ax[0].set(xlabel="Downrange [km]", ylabel="Crossrange [km]",
              title=f"Impact dispersion (N = {args.n})")
    ax[0].axis("equal"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)

    ax[1].plot(t_iip, iip_dr, label="nominal IIP")
    ax[1].axhline(args.destruct_km, color="r", ls="--", label="notional destruct line")
    ax[1].set(xlabel="Time [s]", ylabel="IIP downrange [km]",
              title="Instantaneous impact point during ascent")
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)

    ax[2].semilogx(ns, run_sd, "o-", ms=3)
    ax[2].axhline(sd[0], color="gray", ls="--")
    ax[2].set(xlabel="Number of cases N", ylabel="Downrange 1$\\sigma$ [km]",
              title="Monte Carlo convergence")
    ax[2].grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig("montecarlo.png", dpi=150)
    print("Saved montecarlo.png")


if __name__ == "__main__":
    main()
