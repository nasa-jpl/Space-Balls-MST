"""Numerical regression tests; run with python -m unittest discover -s tests."""
import unittest
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from SpaceBalls.radiation_fluxes_preprocessing import (
    AU, RE, compute_adhya_intersection, compute_shadow_f,
    _shadow_segment_is_blocked, get_solar_incoming_day_hist,
)


class IntersectionTests(unittest.TestCase):
    def test_segment_endpoints_and_direction(self):
        a = np.array([[2, 0, 0], [2, 0, 0], [2, 0, 0], [2, 0, 0]])
        b = np.array([[-4, 0, 0], [4, 0, 0], [-.5, 0, 0], [-1.5, 0, 0]])
        assert_array_equal(_shadow_segment_is_blocked(a, b, 1, 1),
                           [True, False, False, True])

    def test_all_coordinate_axes_and_signed_roots(self):
        # Includes b_x=0, previously singular, and roots behind the observer.
        for axis in np.eye(3):
            det, entry, exit = compute_adhya_intersection(2 * axis, -4 * axis, 1, 1)
            self.assertEqual(det, 1)
            assert_allclose([entry, exit], [0.25, 0.75])
            _, entry, exit = compute_adhya_intersection(2 * axis, 4 * axis, 1, 1)
            assert_allclose([entry, exit], [-0.75, -0.25])

    def test_oblate_axes(self):
        _, entry, exit = compute_adhya_intersection([0, 0, 3], [0, 0, -6], 2, 1)
        assert_allclose([entry, exit], [1 / 3, 2 / 3])

    def test_tangent_and_near_tangent(self):
        a = np.array([[2, 1, 0], [2, 1 + 1e-10, 0], [2, 1 - 1e-10, 0]])
        det, entry, exit = compute_adhya_intersection(a, [-4, 0, 0], 1, 1)
        self.assertEqual(det[0], 0)
        assert_allclose([entry[0], exit[0]], [0.5, 0.5])
        self.assertLess(det[1], 0)
        self.assertTrue(np.isnan(entry[1]) and np.isnan(exit[1]))
        self.assertGreater(det[2], 0)
        self.assertLess(entry[2], exit[2])

    def test_surface_and_interior_origins(self):
        for a, b, expected in [([1, 0, 0], [1, 0, 0], [-2, 0]),
                               ([1, 0, 0], [-1, 0, 0], [0, 2]),
                               ([1, 0, 0], [0, 1, 0], [0, 0]),
                               ([0, 0, 0], [1, 0, 0], [-1, 1])]:
            _, entry, exit = compute_adhya_intersection(a, b, 1, 1)
            assert_allclose([entry, exit], expected)

    def test_near_surface_small_root(self):
        delta = 1e-10
        _, entry, exit = compute_adhya_intersection([1 + delta, 0, 0], [-1, 0, 0], 1, 1)
        assert_allclose(entry, delta, rtol=1e-6, atol=0)
        assert_allclose(exit, 2 + delta)

    def test_large_distance_tangent(self):
        det, entry, exit = compute_adhya_intersection([1e8, 1, 0], [-1e8, 0, 0], 1, 1)
        self.assertEqual(det, 0)
        assert_allclose([entry, exit], [1, 1])

    def test_random_intersections_lie_on_ellipsoid(self):
        rng = np.random.default_rng(2409)
        axes = np.array([6378., 6378., 6357.])
        a = rng.normal(size=(500, 3))
        a *= (2 * axes) / np.linalg.norm(a, axis=-1)[:, None]
        b = -a + rng.normal(size=a.shape) * 500
        det, entry, exit = compute_adhya_intersection(a, b, axes[0], axes[2])
        self.assertTrue(np.all(det > 0))
        for t in (entry, exit):
            points = a + t[:, None] * b
            assert_allclose(np.sum((points / axes)**2, axis=-1), 1, atol=3e-14)
        self.assertTrue(np.all(entry < exit))

    def test_invalid_lines(self):
        for a, b, p, q in [([0, 0, 0], [0, 0, 0], 1, 1),
                            ([np.nan, 0, 0], [1, 0, 0], 1, 1),
                            ([0, 0, 0], [1, np.inf, 0], 1, 1),
                            ([0, 0, 0], [1, 0, 0], 0, 1),
                            ([0, 0, 0], [1, 0, 0], 1, -1)]:
            with self.assertRaises(ValueError):
                compute_adhya_intersection(a, b, p, q)


class ShadowTests(unittest.TestCase):
    def setUp(self):
        self.grid = SimpleNamespace(alt_km=0., flattening=0.)
        self.old_errors = np.seterr(divide='raise', invalid='raise', over='raise')

    def tearDown(self):
        np.seterr(**self.old_errors)

    def shadow(self, observers, sun=None, smooth=True, grid=None, **kwargs):
        if sun is None:
            sun = [[AU, 0, 0]]
        return compute_shadow_f(observers, sun, grid or self.grid, smooth, **kwargs)

    def test_alignment_and_axis_singularities(self):
        for direction in np.eye(3):
            positions = np.array([7000 * direction, -7000 * direction])
            for smooth in (False, True):
                assert_array_equal(self.shadow(positions, [AU * direction], smooth), [[1], [0]])

    def test_near_alignment(self):
        positions = [[-7000, 0, 0], [-7000, 1e-13, 0], [-7000, -1e-9, 0],
                     [-7000, 0, 1e-9], [7000, 1e-13, 0]]
        assert_array_equal(self.shadow(positions), [[0], [0], [0], [0], [1]])

    def test_day_umbra_and_penumbra(self):
        assert_array_equal(self.shadow([[7000, 0, 0], [-7000, 0, 0],
                                        [-7000, RE, 0], [-7000, RE + 100, 0]]),
                           [[1], [0], [0.5], [1]])

    def test_annular_eclipse(self):
        assert_array_equal(self.shadow([[-2e6, 0, 0]]), [[0.5]])
        assert_array_equal(self.shadow([[-2e6, 0, 0]], smooth=False), [[0]])

    def test_earth_behind_observer_or_beyond_sun(self):
        assert_array_equal(self.shadow([[7000, 0, 0], [2 * AU, 0, 0]]), [[1], [1]])

    def test_surface_and_inside(self):
        points = [[RE, 0, 0], [-RE, 0, 0], [RE - 1, 0, 0], [0, 0, 0]]
        for smooth in (True, False):
            assert_array_equal(self.shadow(points, smooth=smooth), [[1], [0], [0], [0]])
        # Exact local tangent: visible for a point Sun, partial for a disc.
        assert_array_equal(self.shadow([[0, RE, 0]], [[AU, RE, 0]], False), [[1]])
        assert_array_equal(self.shadow([[0, RE, 0]], [[AU, RE, 0]], True), [[0.5]])

    def test_point_sun_tangent(self):
        sun = [[AU, RE, 0]]
        points = [[-7000, RE, 0], [-7000, RE + 1e-5, 0], [-7000, RE - 1e-5, 0]]
        assert_array_equal(self.shadow(points, sun, False), [[0], [1], [0]])

    def test_surface_tangents_in_arbitrary_directions(self):
        rng = np.random.default_rng(610)
        axes = np.array([RE, RE, RE * .99])
        origins = rng.normal(size=(1000, 3))
        origins /= np.linalg.norm(origins, axis=-1)[:, None]
        tangents = np.cross(origins, rng.normal(size=origins.shape))
        tangents *= 1e4 / np.linalg.norm(tangents, axis=-1)[:, None]
        assert_array_equal(_shadow_segment_is_blocked(origins * axes, tangents * axes,
                                                      axes[0], axes[2]), False)

    def test_static_and_history_shapes_even_when_points_equal_times(self):
        points = np.array([[7000, 0, 0], [-7000, 0, 0]])
        sun = [[AU, 0, 0], [-AU, 0, 0]]
        expected = [[1, 0], [0, 1]]
        for shape in (points, points[:, None, :], np.repeat(points[:, None, :], 2, axis=1)):
            assert_array_equal(self.shadow(shape, sun), expected)
        moving = np.array([[[7000, 0, 0], [-7000, 0, 0]]])
        assert_array_equal(self.shadow(moving, sun), [[1, 1]])

    def test_flux_caller_with_equal_point_and_time_counts(self):
        points = np.array([[7000, 0, 0], [-7000, 0, 0]])
        sun = np.array([[AU, 0, 0], [-AU, 0, 0]])
        grid = SimpleNamespace(alt_km=7000 - RE)
        tsi = np.array([1361., 1362.])
        flux = get_solar_incoming_day_hist(sun, tsi, points, grid, self.grid)
        self.assertEqual(flux.shape, (2, 2, 3))
        expected = np.zeros_like(flux)
        expected[0, 0, 0] = -tsi[0] * (AU / (AU - 7000))**2
        expected[1, 1, 0] = tsi[1] * (AU / (AU - 7000))**2
        assert_allclose(flux, expected)

    def test_altitude_and_flattening(self):
        point = [[-7000, RE + 30, 0]]
        assert_array_equal(self.shadow(point, smooth=False), [[1]])
        raised = SimpleNamespace(alt_km=100., flattening=0.)
        assert_array_equal(self.shadow(point, smooth=False, grid=raised), [[0]])
        point = [[-7000, 0, RE - 10]]
        assert_array_equal(self.shadow(point, smooth=False), [[0]])
        oblate = SimpleNamespace(alt_km=0., flattening=1 / 298.257223563)
        assert_array_equal(self.shadow(point, smooth=False, grid=oblate), [[1]])

    def test_spherical_reference_over_orbits(self):
        # Independent angular-disc classification, including annular cases.
        # Sample densely across eclipse boundaries at several altitudes.
        rng = np.random.default_rng(2003)
        for radius in (RE + 1., 7000., 42164., 2e6, 1e7):
            angles = np.concatenate((np.linspace(0, np.pi, 3001),
                                     np.arcsin(RE / radius) + np.linspace(-.02, .02, 2001)))
            azimuth = rng.uniform(0, 2 * np.pi, len(angles))
            observers = radius * np.column_stack((-np.cos(angles),
                np.sin(angles) * np.cos(azimuth), np.sin(angles) * np.sin(azimuth)))
            to_sun = np.array([AU, 0, 0]) - observers
            distance = np.linalg.norm(to_sun, axis=-1)
            separation = np.arctan2(np.linalg.norm(np.cross(-observers, to_sun), axis=-1),
                                    np.sum(-observers * to_sun, axis=-1))
            earth_angle = np.arcsin(RE / radius)
            sun_angle = np.arcsin(695700. / distance)
            expected = np.where(separation >= earth_angle + sun_angle, 1.,
                                np.where(separation + sun_angle <= earth_angle, 0., .5))
            assert_array_equal(self.shadow(observers)[:, 0], expected)

    def test_point_sun_against_random_segment_reference(self):
        rng = np.random.default_rng(91)
        axes = np.array([RE, RE, RE * .99])
        observers = rng.normal(size=(2000, 3))
        observers *= 7000 / np.linalg.norm(observers, axis=-1)[:, None]
        sun = np.tile([AU, 0, 0], (2000, 1))
        # Independent quadratic roots for nondegenerate random cases.
        a, b = observers / axes, (sun - observers) / axes
        A, B, C = np.sum(b*b, axis=-1), 2*np.sum(a*b, axis=-1), np.sum(a*a, axis=-1)-1
        det = B*B-4*A*C
        roots = (-B - np.sqrt(np.maximum(det, 0))) / (2*A)
        blocked = (det >= 0) & (roots > 0) & (roots < 1)
        grid = SimpleNamespace(alt_km=0., flattening=.01)
        assert_array_equal(self.shadow(observers[:, None, :], [sun[0]], False, grid)[:, 0],
                           (~blocked).astype(float))

    def test_empty_inputs(self):
        self.assertEqual(self.shadow(np.empty((0, 3))).shape, (0, 1))
        self.assertEqual(self.shadow(np.zeros((2, 3)), np.empty((0, 3))).shape, (2, 0))

    def test_invalid_inputs(self):
        for points, sun in [([1, 2, 3], [[AU, 0, 0]]),
                            ([[7000, 0, 0]], [AU, 0, 0]),
                            (np.zeros((2, 3, 3)), np.zeros((2, 3))),
                            ([[np.nan, 0, 0]], [[AU, 0, 0]]),
                            ([[7000, 0, 0]], [[np.inf, 0, 0]]),
                            ([[AU, 0, 0]], [[AU, 0, 0]]),
                            ([[7000, 0, 0]], [[0, 0, 0]])]:
            with self.assertRaises(ValueError):
                self.shadow(points, sun)
        with self.assertRaises(ValueError):
            self.shadow([[7000, 0, 0]], penumbra_method='not-implemented')
        for flattening in (-.1, 1., np.nan):
            with self.assertRaises(ValueError):
                self.shadow([[7000, 0, 0]], grid=SimpleNamespace(alt_km=0., flattening=flattening))


if __name__ == '__main__':
    unittest.main()
