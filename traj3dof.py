"""
traj3dof.py -- 3DOF point-mass trajectory propagator for a suborbital rocket.

Model
-----
- Round, non-rotating Earth; inverse-square gravity (no J2)
- Exponential atmosphere, constant drag coefficient
- Single stage, constant thrust and Isp
- Ascent: vertical rise -> pitch kick -> gravity turn (thrust along inertial velocity)
- Guided vehicle: thrust held in the launch plane (no weathercocking);
  wind acts through drag only
- Optional constant horizontal wind below WIND_CEILING

Frames
------
Inertial, Earth-centered. Launch site at r0 = [0, 0, Re]:
  x = local East, y = local North, z = Up (at the launch site).
State vector: s = [rx, ry, rz, vx, vy, vz, m]
"""

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

# ---------------- Constants ----------------
MU = 3.986004418e14      # m^3/s^2, Earth gravitational parameter
RE = 6.378137e6          # m, Earth equatorial radius
G0 = 9.80665             # m/s^2, standard gravity
RHO0 = 1.225             # kg/m^3, sea-level density
H_SCALE = 8500.0         # m, atmospheric scale height
WIND_CEILING = 15e3      # m, wind applied only below this altitude


@dataclass
class Vehicle:
    m0: float = 1000.0         # kg, liftoff mass
    m_prop: float = 750.0      # kg, propellant mass
    thrust: float = 25e3       # N
    isp: float = 260.0         # s
    cd: float = 0.30           # drag coefficient (constant)
    diameter: float = 0.60     # m


@dataclass
class Guidance:
    azimuth_deg: float = 90.0  # launch azimuth from North (90 = due East)
    t_vertical: float = 5.0    # s, vertical rise before pitch kick
    t_kick: float = 10.0       # s, duration of pitch-over maneuver
    kick_deg: float = 10.0     # deg, tilt from vertical at end of kick
    yaw_err_deg: float = 0.0   # deg, pitch-kick pointing error (dispersion only)
    pitch_table: tuple = None  # (t, theta_rad) open-loop pitch program; None = gravity turn


@dataclass
class Wind:
    downrange: float = 0.0     # m/s, positive = tailwind
    crossrange: float = 0.0    # m/s, positive = toward +crossrange


# ---------------- Helpers ----------------
def site_frame(azimuth_deg):
    """Unit vectors at the launch site: up, downrange, crossrange (right of downrange)."""
    b = np.radians(azimuth_deg)
    up = np.array([0.0, 0.0, 1.0])
    d = np.array([np.sin(b), np.cos(b), 0.0])     # downrange (East/North components)
    c = np.array([np.cos(b), -np.sin(b), 0.0])    # crossrange, 90 deg clockwise from d
    return up, d, c


def density(h):
    return RHO0 * np.exp(-max(h, 0.0) / H_SCALE)


def ground_coords(r, azimuth_deg):
    """Convert an inertial position to (downrange, crossrange) great-circle distances [m]."""
    up, d, c = site_frame(azimuth_deg)
    u = r / np.linalg.norm(r)
    downrange = RE * np.arctan2(u @ d, u @ up)
    crossrange = RE * np.arcsin(np.clip(u @ c, -1, 1))
    return downrange, crossrange


# ---------------- Equations of motion ----------------
def eom(t, s, veh, gd, wind, phase):
    r, v, m = s[0:3], s[3:6], s[6]
    rn = np.linalg.norm(r)
    h = rn - RE
    up_local = r / rn
    _, d, c = site_frame(gd.azimuth_deg)

    # Air-relative velocity (wind only in the lower atmosphere)
    w = (wind.downrange * d + wind.crossrange * c) if h < WIND_CEILING else np.zeros(3)
    v_rel = v - w
    vr_mag = np.linalg.norm(v_rel)

    # Gravity
    a_grav = -MU * r / rn**3

    # Drag
    area = np.pi * veh.diameter**2 / 4
    if vr_mag > 1e-6:
        a_drag = -0.5 * density(h) * vr_mag * v_rel * veh.cd * area / m
    else:
        a_drag = np.zeros(3)

    # Thrust direction (guidance)
    if phase == "coast":
        a_thrust, mdot = np.zeros(3), 0.0
    else:
        if phase == "vertical":
            u_t = up_local
        elif phase == "kick":
            frac = (t - gd.t_vertical) / gd.t_kick
            k = np.radians(gd.kick_deg) * np.clip(frac, 0, 1)
            _, d_kick, _ = site_frame(gd.azimuth_deg + gd.yaw_err_deg)  # pointing error
            u_t = np.cos(k) * up_local + np.sin(k) * d_kick
        else:
            # Guided vehicle holds its launch plane; pointing error tilts that plane
            _, d_kick, _ = site_frame(gd.azimuth_deg + gd.yaw_err_deg)
            n = np.cross([0.0, 0.0, 1.0], d_kick)          # launch-plane normal
            if gd.pitch_table is None:
                # Gravity turn: thrust along inertial velocity, projected into the plane
                v_p = v - (v @ n) * n
                u_t = v_p / np.linalg.norm(v_p)
            else:
                # Open-loop pitch program: track tabulated angle from local vertical
                tt, th = gd.pitch_table
                theta = np.interp(t, tt, th)
                horiz = np.cross(n, up_local)               # in-plane local horizontal
                u_t = np.cos(theta) * up_local + np.sin(theta) * horiz
        a_thrust = veh.thrust / m * u_t
        mdot = -veh.thrust / (G0 * veh.isp)

    return np.concatenate([v, a_grav + a_drag + a_thrust, [mdot]])


def impact_event(t, s, *args):
    return np.linalg.norm(s[0:3]) - RE
impact_event.terminal = True
impact_event.direction = -1


# ---------------- Propagator ----------------
def simulate(veh=Vehicle(), gd=Guidance(), wind=Wind(), max_step=2.0):
    """Propagate liftoff -> impact. Returns dict with time history and key events."""
    t_burn = veh.m_prop * G0 * veh.isp / veh.thrust
    t1 = gd.t_vertical
    t2 = gd.t_vertical + gd.t_kick
    phases = [("vertical", 0.0, t1), ("kick", t1, t2),
              ("gturn", t2, t_burn), ("coast", t_burn, t_burn + 3000.0)]

    s = np.array([0.0, 0.0, RE, 0.0, 0.0, 0.0, veh.m0])
    T, S = [], []
    burnout_state = None
    for name, ta, tb in phases:
        sol = solve_ivp(eom, (ta, tb), s, args=(veh, gd, wind, name),
                        events=impact_event if name == "coast" else None,
                        rtol=1e-9, atol=1e-6, max_step=max_step)
        T.append(sol.t)
        S.append(sol.y)
        s = sol.y[:, -1]
        if name == "gturn":
            burnout_state = (sol.t[-1], s.copy())

    t = np.concatenate(T)
    S = np.hstack(S)
    r = S[0:3]
    v = S[3:6]
    alt = np.linalg.norm(r, axis=0) - RE
    speed = np.linalg.norm(v, axis=0)
    dr_cr = np.array([ground_coords(r[:, i], gd.azimuth_deg) for i in range(r.shape[1])]).T

    i_apo = np.argmax(alt)
    return {
        "t": t, "state": S, "alt": alt, "speed": speed,
        "downrange": dr_cr[0], "crossrange": dr_cr[1],
        "t_burnout": burnout_state[0], "burnout_state": burnout_state[1],
        "apogee_alt": alt[i_apo], "apogee_time": t[i_apo],
        "impact_time": t[-1], "impact_dr": dr_cr[0, -1], "impact_cr": dr_cr[1, -1],
        "azimuth_deg": gd.azimuth_deg,
    }


def pitch_program(res):
    """Extract the nominal pitch angle from local vertical vs time (powered flight)."""
    t, S = res["t"], res["state"]
    mask = t <= res["t_burnout"]
    r, v = S[0:3, mask], S[3:6, mask]
    vn = np.linalg.norm(v, axis=0)
    cos_th = np.sum(r * v, axis=0) / (np.linalg.norm(r, axis=0) * np.maximum(vn, 1e-9))
    theta = np.arccos(np.clip(cos_th, -1, 1))
    return t[mask], theta


# ---------------- Instantaneous Impact Point ----------------
def iip(r, v, azimuth_deg):
    """Vacuum Keplerian instantaneous impact point for state (r, v).
    Returns (downrange, crossrange) in m, or (nan, nan) if no impact (orbital / degenerate)."""
    rn = np.linalg.norm(r)
    hvec = np.cross(r, v)
    hmag = np.linalg.norm(hvec)
    if hmag < 1e-3:                       # purely vertical: impacts at sub-vehicle point
        return ground_coords(r, azimuth_deg)
    r_hat = r / rn
    t_hat = np.cross(hvec / hmag, r_hat)  # in-plane, direction of motion
    vr = v @ r_hat
    p = hmag**2 / MU
    ecos = p / rn - 1.0
    esin = vr * np.sqrt(p / MU)
    e = np.hypot(ecos, esin)
    nu = np.arctan2(esin, ecos) % (2 * np.pi)
    if e < 1e-12:
        return np.nan, np.nan
    c = (p / RE - 1.0) / e
    if abs(c) > 1:                        # conic never reaches Earth's surface
        return np.nan, np.nan
    nu_imp = 2 * np.pi - np.arccos(c)     # descending crossing of r = RE
    psi = nu_imp - nu                     # range angle still to travel
    if psi < 0:
        return np.nan, np.nan
    r_imp = RE * (np.cos(psi) * r_hat + np.sin(psi) * t_hat)
    return ground_coords(r_imp, azimuth_deg)


# ---------------- Run nominal ----------------
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = simulate()
    bo = res["burnout_state"]
    print(f"Burnout:  t = {res['t_burnout']:.1f} s, "
          f"alt = {(np.linalg.norm(bo[0:3]) - RE) / 1e3:.1f} km, "
          f"speed = {np.linalg.norm(bo[3:6]):.0f} m/s")
    print(f"Apogee:   t = {res['apogee_time']:.1f} s, alt = {res['apogee_alt'] / 1e3:.1f} km")
    print(f"Impact:   t = {res['impact_time']:.1f} s, "
          f"downrange = {res['impact_dr'] / 1e3:.1f} km, crossrange = {res['impact_cr'] / 1e3:.2f} km")

    # Sanity check: rocket-equation ideal delta-V vs achieved burnout speed
    veh = Vehicle()
    dv_ideal = veh.isp * G0 * np.log(veh.m0 / (veh.m0 - veh.m_prop))
    print(f"Ideal dV = {dv_ideal:.0f} m/s, losses (gravity + drag + steering) "
          f"= {dv_ideal - np.linalg.norm(bo[3:6]):.0f} m/s")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(res["downrange"] / 1e3, res["alt"] / 1e3)
    ax[0].set(xlabel="Downrange [km]", ylabel="Altitude [km]", title="Nominal trajectory")
    ax[0].grid(alpha=0.3)
    ax[1].plot(res["t"], res["speed"])
    ax[1].axvline(res["t_burnout"], ls="--", c="gray", label="burnout")
    ax[1].set(xlabel="Time [s]", ylabel="Speed [m/s]", title="Inertial speed")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("nominal.png", dpi=150)
    print("Saved nominal.png")
