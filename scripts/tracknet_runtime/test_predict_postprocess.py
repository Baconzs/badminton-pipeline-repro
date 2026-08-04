import os
import sys
import unittest


RUNTIME_DIR = os.path.dirname(os.path.abspath(__file__))
if RUNTIME_DIR not in sys.path:
    sys.path.insert(0, RUNTIME_DIR)

from predict import suppress_static_prediction_locks


def make_prediction(points):
    return {
        "Frame": list(range(len(points))),
        "X": [point[0] for point in points],
        "Y": [point[1] for point in points],
        "Visibility": [1] * len(points),
        "metadata": "preserved",
    }


class StaticPredictionLockTests(unittest.TestCase):
    def test_slow_moving_flight_is_not_extended_as_static_lock(self):
        # Every four-frame subsection is compact enough to trigger the old
        # filter.  Across eight frames, however, the shuttle has clearly left
        # the original centre and must remain visible.
        points = [(500 + 5 * frame, 220 + 2 * frame) for frame in range(18)]
        output = suppress_static_prediction_locks(
            make_prediction(points), window=4, disp=10, min_y=80, preroll=2
        )
        self.assertTrue(all(int(value) == 1 for value in output["Visibility"]))

    def test_four_frame_compact_patch_is_not_enough_evidence(self):
        points = [(500 + frame % 2, 400) for frame in range(4)]
        output = suppress_static_prediction_locks(
            make_prediction(points), window=4, disp=10, min_y=80, preroll=2
        )
        self.assertEqual(output["Visibility"], [1, 1, 1, 1])

    def test_persistent_fixed_response_is_removed(self):
        points = [(500 + frame % 2, 400 + frame % 3) for frame in range(20)]
        output = suppress_static_prediction_locks(
            make_prediction(points), window=4, disp=10, min_y=80, preroll=2
        )
        self.assertTrue(all(int(value) == 0 for value in output["Visibility"]))
        self.assertEqual(output["metadata"], "preserved")

    def test_fixed_lock_does_not_swallow_later_flight(self):
        lock = [(500 + frame % 2, 400 + frame % 3) for frame in range(10)]
        flight = [(530 + 8 * frame, 420 + 4 * frame) for frame in range(10)]
        output = suppress_static_prediction_locks(
            make_prediction(lock + flight),
            window=4,
            disp=10,
            min_y=80,
            preroll=0,
        )
        self.assertTrue(all(int(output["Visibility"][frame]) == 0 for frame in range(10)))
        self.assertTrue(all(int(output["Visibility"][frame]) == 1 for frame in range(10, 20)))


if __name__ == "__main__":
    unittest.main()
