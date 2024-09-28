import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os, sys
import math

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.filepath import ABSOLUTE_PATHGRF
from tqdm.auto import tqdm


def coupled_diffusion_equation(t, uv, Du, Dv, dx):
    """1D FN reaction diffusion equation

    Args:
        t (_type_): timestep
        uv (_type_): u and v
        Du (_type_): coefficient
        Dv (_type_): coefficient
        dx (_type_): space step

    Returns:
        _type_: right side of couple PDE equation
    """
    Nx = uv.shape[0] // 2
    u, v = uv[:Nx], uv[Nx:]
    u_xx = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx**2
    v_xx = (np.roll(v, -1) - 2 * v + np.roll(v, 1)) / dx**2
    u_xx[0], u_xx[-1] = 0, (u[-3] - u[-1]) / (2 * dx**2)
    v_xx[0], v_xx[-1] = 0, (v[-3] - v[-1]) / (2 * dx**2)
    dUdt = Du * u_xx + u - u**3 - v + 0.1
    dVdt = Dv * v_xx + (u - v) * 0.25
    return np.concatenate((dUdt, dVdt), axis=-1)


def coupled_diffusion_equation_u(t, u, Du, v_tot, dx, t_end):
    """FN reaction diffusion equation, give v to solve u

    Args:
        t (_type_): timestep
        u (_type_): u
        Du (_type_): coefficient
        v_tot (_type_): given v in all timestep
        dx (_type_): space step
        t_end (_type_): end of time

    Returns:
        _type_: right side of single PDE u
    """
    n_t = v_tot.shape[0]
    t_index = t / t_end * (n_t - 1)
    t_index_up = math.ceil(t_index)
    t_index_low = math.floor(t_index)
    v = v_tot[t_index_low] + (t_index - t_index_low) * (v_tot[t_index_up] - v_tot[t_index_low])
    u_xx = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx**2
    u_xx[0], u_xx[-1] = 0, (u[-3] - u[-1]) / (2 * dx**2)
    dUdt = Du * u_xx + u - u**3 - v + 0.1
    return dUdt


def coupled_diffusion_equation_v(t, v, Dv, u_tot, dx, t_end):
    """FN reaction diffusion equation, give u to solve v

    Args:
        t (_type_): timestep
        v (_type_): v
        Du (_type_): coefficient
        u_tot (_type_): given u in all timestep
        dx (_type_): space step
        t_end (_type_): end of time

    Returns:
        _type_: right side of single PDE v
    """
    n_t = u_tot.shape[0]
    t_index = t / t_end * (n_t - 1)
    t_index_up = math.ceil(t_index)
    t_index_low = math.floor(t_index)
    u = u_tot[t_index_low] + (t_index - t_index_low) * (u_tot[t_index_up] - u_tot[t_index_low])
    v_xx = (np.roll(v, -1) - 2 * v + np.roll(v, 1)) / dx**2
    v_xx[0], v_xx[-1] = 0, (v[-3] - v[-1]) / (2 * dx**2)
    dVdt = Dv * v_xx + (u - v) * 0.25
    return dVdt


def gen_u_from_v(n_sample, L=1, N=20, t_end=5, Du=0.01):
    """generate u from given v

    Args:
        n_sample (_type_): number of samples.
        L (int, optional):  Space length.
        N (int, optional):  Number of spatial discrete grid points.
        t_end (int, optional): end of time. Defaults to 5.
        Du (float, optional): coefficient. Defaults to 0.01.
    """
    v_tot_lis = []
    u_tot_lis = []

    t_span = (0, t_end)
    dx = L / (N - 1)
    grf = GRF(N=N)
    for i in tqdm(range(n_sample), desc="sampling loop time step", total=n_sample):
        initial_u = grf.random(1).flatten()
        v_tot = grf.random(10)
        sol = solve_ivp(
            coupled_diffusion_equation_u,
            t_span,
            y0=initial_u,
            args=(Du, v_tot, dx, t_end),
            t_eval=np.linspace(0, t_end, 10),
        )
        u_tot = sol.y  # N, t
        v_tot_lis.append(v_tot)
        u_tot_lis.append(u_tot.T)

    np.save(os.path.join(ABSOLUTE_PATH, "data/reaction_diffusion_u_from_v_u.npy"), np.array(u_tot_lis))
    np.save(os.path.join(ABSOLUTE_PATH, "data/reaction_diffusion_u_from_v_v.npy"), np.array(v_tot_lis))
    #


def gen_v_from_u(n_sample, L=1, N=20, t_end=5, Dv=0.05):
    """generate v from given u

    Args:
        n_sample (_type_): number of samples.
        L (int, optional):  Space length.
        N (int, optional):  Number of spatial discrete grid points.
        t_end (int, optional): end of time. Defaults to 5.
        Dv (float, optional): coefficient. Defaults to 0.05.
    """
    v_tot_lis = []
    u_tot_lis = []

    t_span = (0, t_end)
    dx = L / (N - 1)
    grf = GRF(N=N)
    for i in tqdm(range(n_sample), desc="sampling loop time step", total=n_sample):
        initial_v = grf.random(1).flatten()
        u_tot = grf.random(10)
        sol = solve_ivp(
            coupled_diffusion_equation_v,
            t_span,
            y0=initial_v,
            args=(Dv, u_tot, dx, t_end),
            t_eval=np.linspace(0, t_end, 10),
        )
        v_tot = sol.y  # N, t
        v_tot_lis.append(v_tot.T)
        u_tot_lis.append(u_tot)

    np.save(os.path.join(ABSOLUTE_PATH, "data/reaction_diffusion_v_from_u_u.npy"), np.array(u_tot_lis))
    np.save(os.path.join(ABSOLUTE_PATH, "data/reaction_diffusion_v_from_u_v.npy"), np.array(v_tot_lis))


def gen_uv(n_sample, L=1, N=20, t_end=5, Du=0.01, Dv=0.05):
    """generate uv

    Args:
        n_sample (_type_): number of samples.
        L (int, optional):  Space length.
        N (int, optional):  Number of spatial discrete grid points.
        t_end (int, optional): end of time. Defaults to 5.
        Du (float, optional): coefficient. Defaults to 0.01.
        Dv (float, optional): coefficient. Defaults to 0.05.
    """
    t_span = (0, t_end)
    dx = L / (N - 1)
    grf = GRF(N=N)
    uv_lis = []
    for i in tqdm(range(n_sample), desc="sampling loop time step", total=n_sample):
        initial_u = grf.random(1).flatten()
        initial_v = grf.random(1).flatten()
        initial_uv = np.concatenate((initial_u, initial_v), axis=0).flatten()
        sol = solve_ivp(
            coupled_diffusion_equation, t_span, y0=initial_uv, args=(Du, Dv, dx), t_eval=np.linspace(0, t_end, 10)
        )
        uv_lis.append(sol.y)
    np.save(os.path.join(ABSOLUTE_PATH, "data/reaction_diffusion_uv.npy"), np.array(uv_lis))


# gen_u_from_v(10000)
# gen_v_from_u(10000)
gen_uv(10000)
