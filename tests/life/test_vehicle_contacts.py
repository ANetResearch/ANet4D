import unittest
from unittest.mock import patch

import numpy as np

from awr.life.civil.grounded import support_mask


class ContactSupport(unittest.TestCase):
    def test_hole_inside_supported_chassis_corners_rejects_route(self):
        class Surface:
            def supported(self, points, profile):
                # Corners and centre are sound; a tyre lies over a mesh hole.
                return np.linalg.norm(np.asarray(points) - [0.9, 0.6], axis=-1) > 0.05

        points = np.array([[0.0, 0.0], [0.01, 0.0]])
        with patch("awr.life.civil.contacts.vehicle_contacts", return_value=np.array([[0.9, 0.6]])):
            result = support_mask(Surface(), points, np.zeros(2), [4.0, 2.0, 1.5], "vehicle")
        self.assertFalse(result.any())
