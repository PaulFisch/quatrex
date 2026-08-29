"""Exactness tests for the private adaptive P1 projected convolution."""

import numpy as np

from studies import _adaptive_p1_fct as F


def _direct_convolution(f, g, z):
    """Independent piecewise Gauss rule split at every P1 breakpoint."""
    nodes, weights = np.polynomial.legendre.leggauss(4)
    fv = f.mesh.vertices
    gv = g.mesh.vertices
    lo = max(float(fv[0]), z - float(gv[-1]))
    hi = min(float(fv[-1]), z - float(gv[0]))
    if hi <= lo:
        return 0.0
    cuts = np.unique(np.concatenate((
        [lo, hi], fv[(fv > lo) & (fv < hi)],
        (z - gv)[(z - gv > lo) & (z - gv < hi)])))
    result = 0.0
    for left, right in zip(cuts[:-1], cuts[1:]):
        x = 0.5 * ((right - left) * nodes + right + left)
        result += 0.5 * (right - left) * np.sum(
            weights * F.evaluate(f, x) * F.evaluate(g, z - x))
    return result


def _project_direct(f, g, mesh):
    nodes, weights = np.polynomial.legendre.leggauss(5)
    sum_breaks = np.unique(
        np.add.outer(f.mesh.vertices, g.mesh.vertices).ravel())
    levels = {}
    for level in mesh.levels:
        idx = mesh.level_indices(level)
        start, stop = int(idx.min()), int(idx.max()) + 1
        data = np.zeros((stop - start, 2))
        h = mesh.base_h / 2**level
        for index in idx:
            left, right = index * h, (index + 1) * h
            cuts = np.concatenate(([left], sum_breaks[
                (sum_breaks > left) & (sum_breaks < right)], [right]))
            for a, b in zip(cuts[:-1], cuts[1:]):
                z = 0.5 * ((b - a) * nodes + b + a)
                vals = np.array([_direct_convolution(f, g, zz) for zz in z])
                phi0 = np.full_like(z, 1.0 / np.sqrt(h))
                phi1 = np.sqrt(12.0) * (
                    z - 0.5 * (left + right)) / h**1.5
                data[index - start, 0] += (
                    0.5 * (b - a) * np.sum(weights * vals * phi0))
                data[index - start, 1] += (
                    0.5 * (b - a) * np.sum(weights * vals * phi1))
        levels[level] = F.Sequence(start, data)
    return F.P1Field(mesh, levels)


def _meshes():
    inp = F.DyadicMesh.refined(
        1.0, 0, 4,
        lambda left, right: 2 if 1.0 <= left and right <= 3.0 else (
            1 if 0.0 <= left and right <= 3.0 else 0))
    out = F.DyadicMesh.refined(
        1.0, 0, 8,
        lambda left, right: 3 if 2.0 <= left and right <= 5.0 else (
            1 if 1.0 <= left and right <= 7.0 else 0))
    return inp, out


def test_refine_and_coarsen_are_exact_for_p1_coefficients():
    rng = np.random.default_rng(4)
    coarse = F.Sequence(-2, rng.normal(size=(7, 2, 3)))
    fine = F.refine_p1(coarse, 3)
    got = F.coarsen_p1(fine, 3)
    assert got.start == coarse.start
    assert np.allclose(got.data, coarse.data, rtol=2e-14, atol=2e-14)


def test_all_level_cases_match_direct_projected_convolution():
    inp, out = _meshes()

    def f(x):
        x = np.asarray(x)
        return np.where((x >= 0.0) & (x <= 4.0),
                        0.3 + 0.2 * x + 0.15 * np.abs(x - 1.75), 0.0)

    def g(x):
        x = np.asarray(x)
        return np.where((x >= 0.0) & (x <= 4.0),
                        0.7 - 0.08 * x + 0.1 * np.abs(x - 2.25), 0.0)

    ff = F.P1Field.from_callable(inp, f)
    gg = F.P1Field.from_callable(inp, g)
    got = F.projected_convolution(ff, gg, out)
    combined = F.projected_convolution_combined(ff, gg, out)
    want = _project_direct(ff, gg, out)
    for level in out.levels:
        idx = out.level_indices(level)
        a = got.levels[level].sample(idx)
        c = combined.levels[level].sample(idx)
        b = want.levels[level].sample(idx)
        assert np.allclose(a, b, rtol=3e-10, atol=3e-10), (level, a, b)
        assert np.allclose(c, b, rtol=3e-10, atol=3e-10), (level, c, b)


def test_generic_bilinear_callback_matches_componentwise_reference():
    mesh = F.DyadicMesh.refined(
        1.0, 0, 4, lambda left, right: 1 if 1 <= left < 3 else 0)
    out = F.DyadicMesh.refined(
        1.0, 0, 8, lambda left, right: 1 if 2 <= left < 6 else 0)
    a = F.P1Field.from_callable(
        mesh, lambda x: np.stack((1.0 + x, 0.3 - 0.2 * x), axis=-1))
    b = F.P1Field.from_callable(
        mesh, lambda x: np.stack((0.5 + 0.1 * x, -0.4 + x), axis=-1))

    def dot_ring(x, y):
        return np.sum(x * y, axis=-1)

    got = F.projected_convolution(a, b, out, dot_ring)
    combined = F.projected_convolution_combined(a, b, out, dot_ring)
    want = None
    for component in range(2):
        af = F.P1Field(mesh, {l: F.Sequence(s.start, s.data[..., component])
                              for l, s in a.levels.items()})
        bf = F.P1Field(mesh, {l: F.Sequence(s.start, s.data[..., component])
                              for l, s in b.levels.items()})
        term = F.projected_convolution(af, bf, out)
        if want is None:
            want = term
        else:
            want = F.P1Field(out, {
                l: F._sum_sequences([want.levels[l], term.levels[l]])
                for l in out.levels})
    for level in out.levels:
        idx = out.level_indices(level)
        assert np.allclose(got.levels[level].sample(idx),
                           want.levels[level].sample(idx),
                           rtol=3e-11, atol=3e-11)
        assert np.allclose(combined.levels[level].sample(idx),
                           want.levels[level].sample(idx),
                           rtol=3e-11, atol=3e-11)


def test_combined_noncommutative_callback_matches_pair_expansion():
    mesh, out = _meshes()

    def matrix_a(x):
        x = np.asarray(x)
        result = np.zeros(x.shape + (2, 2))
        result[..., 0, 0] = 1.0 + 0.1 * x
        result[..., 0, 1] = 0.2 - 0.03 * x
        result[..., 1, 0] = -0.1 + 0.04 * x
        result[..., 1, 1] = 0.7 + 0.02 * x
        return result

    def matrix_b(x):
        x = np.asarray(x)
        result = np.zeros(x.shape + (2, 2))
        result[..., 0, 0] = 0.3 + 0.02 * x
        result[..., 0, 1] = -0.2 + 0.05 * x
        result[..., 1, 0] = 0.4 - 0.01 * x
        result[..., 1, 1] = 1.1 - 0.04 * x
        return result

    a = F.P1Field.from_callable(mesh, matrix_a)
    b = F.P1Field.from_callable(mesh, matrix_b)
    direct = F.projected_convolution(a, b, out, np.matmul)
    combined = F.projected_convolution_combined(a, b, out, np.matmul)
    for level in out.levels:
        idx = out.level_indices(level)
        np.testing.assert_allclose(
            combined.levels[level].sample(idx),
            direct.levels[level].sample(idx), rtol=3e-11, atol=3e-11)


def test_projection_preserves_mass_and_first_moment():
    inp = F.DyadicMesh.refined(
        1.0, -2, 2,
        lambda left, right: 3 if -0.5 <= left and right <= 1.0 else 1)
    out = F.DyadicMesh.refined(
        1.0, -4, 4,
        lambda left, right: 3 if -1.0 <= left and right <= 2.0 else 1)
    f = F.P1Field.from_callable(
        inp, lambda x: np.maximum(0.0, 1.0 - (x / 2.0) ** 2))
    g = F.P1Field.from_callable(
        inp, lambda x: np.maximum(0.0, 1.0 - np.abs(x) / 2.0))
    bubble = F.projected_convolution_combined(f, g, out)
    f0, f1 = F.p1_moments(f)
    g0, g1 = F.p1_moments(g)
    b0, b1 = F.p1_moments(bubble)
    np.testing.assert_allclose(b0, f0 * g0, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        b1, f1 * g0 + f0 * g1, rtol=2e-13, atol=2e-13)

    continuous = F.project_continuous(bubble)
    reconstructed = F.P1Field.from_vertices(out, continuous.values)
    c0, c1 = F.p1_moments(reconstructed)
    np.testing.assert_allclose(c0, b0, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(c1, b1, rtol=2e-13, atol=2e-13)


def test_fc3_ring_matrix_path_and_diagonal_keldysh_symmetry():
    inp = F.DyadicMesh.refined(
        1.0, -2, 2,
        lambda left, right: 2 if -1.0 <= left and right <= 1.0 else 1)
    out = F.DyadicMesh.refined(
        1.0, -4, 4,
        lambda left, right: 2 if -2.0 <= left and right <= 2.0 else 1)

    def carrier(x):
        x = np.asarray(x)
        v = np.stack((1.0 + 0.05 * x, 0.4 - 0.03 * x), axis=-1)
        return np.einsum("...i,...j->...ij", v, v)

    green = F.P1Field.from_callable(inp, lambda x: -1j * carrier(x))
    phi = np.array([[[0.8, 0.2], [0.2, -0.1]],
                    [[0.1, -0.3], [-0.3, 0.7]]])

    def ring(a, b):
        # Same index algebra as quatrex.phonon.bubble.ring_contract, kept
        # local so this unit test does not initialise MPI.
        return np.einsum(
            "ace,wcb,wed,Jdb->waJ", phi, a, b, phi, optimize=True)

    pairwise = F.projected_convolution(green, green, out, ring)
    combined = F.projected_convolution_combined(green, green, out, ring)
    for level in out.levels:
        idx = out.level_indices(level)
        a = combined.levels[level].sample(idx)
        b = pairwise.levels[level].sample(idx)
        np.testing.assert_allclose(a, b, rtol=4e-11, atol=4e-11)
        sigma = 0.5j * a
        np.testing.assert_allclose(
            sigma + sigma.swapaxes(-1, -2).conj(), 0.0,
            rtol=0.0, atol=2e-11)
