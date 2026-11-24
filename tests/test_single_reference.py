"""Small synthetic checks for the single-reference correction interface.

Run from the repository root: python -B -m unittest discover -s tests -v
"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.correction import time_correction
from src.pipeline import apply_correction
from src.utils import CyTOFContext, get_config


class SingleReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = {
            "data_dir": self.temp.name,
            "output_dir": str(Path(self.temp.name) / "output"),
            "ref_plate": "Reference",
            "anchors": {"Example": "anchor"},
            "meta_cols": ["Time", "panel", "plate", "sample_name", "dataset"],
            "celltypes": ["selected", "other"],
            "celltype_gating": [],
            "celltype2correct": ["selected"],
            "stable_marker": "ReferenceSignal",
            "true_markers": ["Marker"],
        }
        n = 80
        self.frame = pd.DataFrame({
            "Time": np.arange(n),
            "panel": "ExamplePanel",
            "plate": "Reference",
            "sample_name": "anchor",
            "dataset": "Example",
            "celltypes": ["selected"] * 60 + ["other"] * 20,
            "ReferenceSignal": np.full(n, 10.0, dtype=np.float32),
            "Marker": np.linspace(1, 20, n, dtype=np.float32),
        })

    def test_custom_single_reference_and_constant_signal(self):
        ctx = CyTOFContext(get_config(config_dict=self.config))
        original = self.frame.copy(deep=True)
        result = time_correction(self.frame, "ExamplePanel", ctx)
        pd.testing.assert_frame_equal(result, original)
        pd.testing.assert_frame_equal(self.frame, original)

    def test_invalid_reference_types_rejected_at_each_entry(self):
        for value in (None, "", "  ", [], ["ReferenceSignal"], ["A", "B"], 42):
            with self.subTest(stable_marker=value):
                config = dict(self.config, stable_marker=value)
                with self.assertRaisesRegex(ValueError, "non-empty string"):
                    get_config(config_dict=config)
                self.assertFalse(Path(config["output_dir"]).exists())
                with self.assertRaisesRegex(ValueError, "non-empty string"):
                    CyTOFContext(config)
                ctx = SimpleNamespace(**config)
                with self.assertRaisesRegex(ValueError, "non-empty string"):
                    time_correction(self.frame, "ExamplePanel", ctx)
                original = self.frame.copy(deep=True)
                with self.assertRaisesRegex(ValueError, "non-empty string"):
                    apply_correction(self.frame, ["Marker"], "anchor", ctx)
                pd.testing.assert_frame_equal(self.frame, original)

    def test_missing_reference_column_rejected(self):
        ctx = CyTOFContext(dict(self.config, stable_marker="MissingReference"))
        with self.assertRaisesRegex(ValueError, "MissingReference.*absent"):
            time_correction(self.frame, "ExamplePanel", ctx)

    def test_skipping_time_correction_keeps_valid_string_required(self):
        ctx = CyTOFContext(dict(self.config, celltype2correct=[]))
        pd.testing.assert_frame_equal(
            time_correction(self.frame, "ExamplePanel", ctx), self.frame
        )
        ctx.stable_marker = []
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            time_correction(self.frame, "ExamplePanel", ctx)

    def test_missing_time_preserves_existing_skip_behavior(self):
        ctx = CyTOFContext(self.config)
        frame = self.frame.drop(columns="Time")
        pd.testing.assert_frame_equal(
            time_correction(frame, "ExamplePanel", ctx), frame
        )


if __name__ == "__main__":
    unittest.main()
