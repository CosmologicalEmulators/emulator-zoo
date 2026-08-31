"""Velocileptors-style bias-basis decomposition of undamped Folps EFT."""

import numpy as np


K_GRID = np.concatenate((
    np.asarray([5.0e-4]),
    np.geomspace(1.5e-3, 2.5e-2, 10),
    np.arange(3.0e-2, 5.0e-1 + 5.0e-12, 1.0e-2),
))
ELLS = (0, 2, 4)
LOOP_BIAS_COUNT = 12


def _legendre(ell, mu):
    if ell == 0:
        return np.ones_like(mu)
    if ell == 2:
        return 0.5 * (3.0 * mu**2 - 1.0)
    if ell == 4:
        return 0.125 * (35.0 * mu**4 - 30.0 * mu**2 + 3.0)
    raise ValueError(f"Unsupported multipole {ell}")


def quadrature(nmu=6):
    nodes, weights = np.polynomial.legendre.leggauss(2 * nmu)
    return nodes[nmu:], 0.5 * (weights[nmu:] + weights[nmu - 1 :: -1])


def _column(table, index):
    return np.asarray(table[index])[:, None]


def _loop_basis_pkmu(k, mu, table):
    """Return the 12 Folps loop columns before IR mixing and projection."""
    pkl = _column(table, 0)
    ratio = _column(table, 1)
    ploop_dd, ploop_dt, ploop_tt = (_column(table, index) for index in range(2, 5))
    pb1b2, pb1bs, pb22, pb2bs, pbsbs = (_column(table, index) for index in range(5, 10))
    sigma23pkl = _column(table, 10)
    pb2t, pbst = _column(table, 11), _column(table, 12)
    i1, i21, i22, i32, i33 = (_column(table, index) for index in range(13, 18))
    d21, d22, d32, d33, d42, d43, d44 = (_column(table, index) for index in range(18, 25))
    d31, d41 = _column(table, 25), _column(table, 26)
    i1b2, i21b2, i22b2 = (_column(table, index) for index in range(27, 30))
    i1bs, i21bs, i22bs = (_column(table, index) for index in range(30, 33))
    sigma2w = np.asarray(table[33])
    f0 = np.asarray(table[-1])

    mu2 = mu**2
    mu4 = mu2**2
    mu6 = mu4 * mu2
    mu8 = mu4**2
    result = np.zeros((k.shape[0], mu.shape[1], LOOP_BIAS_COUNT), dtype=np.result_type(pkl, f0))

    # Bias-independent, b1, and b1^2 terms.
    result[:, :, 0] = (
        f0**2 * mu4 * ploop_tt
        + f0**3 * (mu4 * i32 + mu6 * i33)
        + f0**4 * (mu2 * d41 + mu4 * d42 + mu6 * d43 + mu8 * d44)
    )
    result[:, :, 1] = (
        2.0 * f0 * mu2 * ploop_dt
        + f0**2 * (mu2 * i21 + mu4 * i22)
        + f0**3 * (mu2 * d31 + mu4 * d32 + mu6 * d33)
    )
    result[:, :, 2] = (
        ploop_dd
        + f0 * mu2 * i1
        + f0**2 * (mu2 * d21 + mu4 * d22)
    )

    # b2 and tidal-bias terms.
    result[:, :, 3] = 2.0 * f0 * mu2 * pb2t + 0.5 * f0**2 * (mu2 * i21b2 + mu4 * i22b2)
    result[:, :, 4] = 2.0 * pb1b2 + 0.5 * f0 * mu2 * i1b2
    result[:, :, 5] = pb22
    result[:, :, 6] = 2.0 * f0 * mu2 * pbst + 0.5 * f0**2 * (mu2 * i21bs + mu4 * i22bs)
    result[:, :, 7] = 2.0 * pb1bs + 0.5 * f0 * mu2 * i1bs
    result[:, :, 8] = 2.0 * pb2bs
    result[:, :, 9] = pbsbs
    result[:, :, 10] = 2.0 * f0 * mu2 * ratio * sigma23pkl
    result[:, :, 11] = 2.0 * sigma23pkl

    # Perturbative GTNS contribution retained by native Folps EFT.
    gtns_prefactor = -(k * mu * f0) ** 2 * sigma2w
    result[:, :, 0] += gtns_prefactor * f0**2 * mu4 * ratio**2 * pkl
    result[:, :, 1] += gtns_prefactor * 2.0 * f0 * mu2 * ratio * pkl
    result[:, :, 2] += gtns_prefactor * pkl
    return result


def build_pkmu_basis(table, table_now, nmu=6):
    """Build AP-free, IR-resummed P11/loop/counterterm basis tables."""
    k = np.asarray(table[0])[:, None]
    mu_values, weights = quadrature(nmu)
    mu = mu_values[None, :]
    pkl, pkl_now = _column(table, 1), _column(table_now, 1)
    ratio = _column(table, 2)
    f0 = np.asarray(table[-1])
    sigma2_now, delta_sigma2_now = np.asarray(table_now[-3]), np.asarray(table_now[-2])
    sigma2_total = (
        (1.0 + f0 * mu**2 * (2.0 + f0)) * sigma2_now
        + (f0 * mu) ** 2 * (mu**2 - 1.0) * delta_sigma2_now
    )
    exponential = np.exp(-k**2 * sigma2_total)

    p_ir = pkl_now + exponential * (pkl - pkl_now) * (1.0 + k**2 * sigma2_total)
    p11 = np.stack((
        np.broadcast_to(p_ir, p_ir.shape),
        ratio * mu**2 * p_ir,
        ratio**2 * mu**4 * p_ir,
    ), axis=-1)

    loop_wiggle = _loop_basis_pkmu(k, mu, table[1:])
    loop_now = _loop_basis_pkmu(k, mu, table_now[1:])
    loop = exponential[:, :, None] * loop_wiggle + (1.0 - exponential[:, :, None]) * loop_now

    p_ct_ir = exponential * pkl + (1.0 - exponential) * pkl_now
    ct_standard = np.stack((
        k**2 * p_ct_ir,
        k**2 * mu**2 * p_ct_ir,
        k**2 * mu**4 * p_ct_ir,
    ), axis=-1)

    def nlo(table_values):
        p = _column(table_values, 0)
        r = _column(table_values, 1)
        sigma2w = np.asarray(table_values[33])
        base = (k * mu * f0) ** 4 * sigma2w**2 * p
        return np.stack((base, r * mu**2 * base, r**2 * mu**4 * base), axis=-1)

    nlo_wiggle = nlo(table[1:])
    nlo_now = nlo(table_now[1:])
    ct_nlo = exponential[:, :, None] * nlo_wiggle + (1.0 - exponential[:, :, None]) * nlo_now
    ct = np.concatenate((ct_standard, ct_nlo), axis=-1)
    return mu_values, weights, p11, loop, ct


def project_basis(table, table_now, nmu=6, ells=ELLS):
    mu, weights, p11_pkmu, loop_pkmu, ct_pkmu = build_pkmu_basis(table, table_now, nmu=nmu)
    result = {}
    for ell in ells:
        projection = weights * (2 * ell + 1) * _legendre(ell, mu)
        p11 = np.einsum("kmc,m->kc", p11_pkmu, projection)
        loop = np.einsum("kmc,m->kc", loop_pkmu, projection)
        ct = np.einsum("kmc,m->kc", ct_pkmu, projection)
        result[ell] = np.concatenate((p11, loop, ct), axis=1)
    return result


def bias_coefficients(nuisance):
    """Return coefficients for the 21 emulated basis columns."""
    b1, b2, bs, b3, alpha0, alpha2, alpha4, ctilde, _, _, _, f0 = nuisance
    p11 = np.asarray([b1**2, 2.0 * b1 * f0, f0**2])
    loop = np.asarray([
        1.0, b1, b1**2, b2, b1 * b2, b2**2,
        bs, b1 * bs, b2 * bs, bs**2, b3, b1 * b3,
    ])
    ct = np.asarray([
        alpha0, alpha2, alpha4,
        ctilde * b1**2, 2.0 * ctilde * b1 * f0, ctilde * f0**2,
    ])
    return np.concatenate((p11, loop, ct))


def stochastic_multipoles(k, nuisance):
    _, _, _, _, _, _, _, _, alpha_shot0, alpha_shot2, pshot, _ = nuisance
    k = np.asarray(k)
    return {
        0: pshot * (alpha_shot0 + alpha_shot2 * k**2 / 3.0),
        2: pshot * (2.0 / 3.0) * alpha_shot2 * k**2,
        4: np.zeros_like(k),
    }


def contract_basis(basis, k, nuisance):
    coefficients = bias_coefficients(nuisance)
    stochastic = stochastic_multipoles(k, nuisance)
    return np.stack([basis[ell] @ coefficients + stochastic[ell] for ell in ELLS])
