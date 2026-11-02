import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from awr.life.civil.surface import SurfaceError, SurfaceMap


class SurfaceResolution(unittest.TestCase):
    def test_metric_coordinates_and_holes_on_quarter_metre_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "meta.json").write_text(
                json.dumps(dict(schema="awr.life.support.v1", x0=-10, y0=20, nx=40, ny=40, cell_m=0.25))
            )
            y, x = np.mgrid[:40, :40]
            z = 0.1 * (-10 + (x + 0.5) * 0.25) + 0.2 * (20 + (y + 0.5) * 0.25)
            z[20, 20] = np.nan
            np.save(p / "ground.npy", z)
            for profile in ("person", "vehicle"):
                np.save(p / f"walk_{profile}.npy", np.ones(z.shape, bool))
            s = SurfaceMap(p)
            xy = np.array([[-8.2, 22.7], [-3.4, 26.1]])
            np.testing.assert_allclose(s.sample(xy), xy[:, 0] * 0.1 + xy[:, 1] * 0.2)
            self.assertTrue(s.supported(xy).all())
            self.assertEqual(s.width_m, 10)
            with self.assertRaises(SurfaceError):
                s.sample([[-4.875, 25.125]])
            self.assertFalse(s.supported([[-4.875, 25.125]])[0])
