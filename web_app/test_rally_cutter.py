import csv
import tempfile
import unittest
from pathlib import Path

from web_app.rally_cutter import read_rally_segments


class RallySegmentTests(unittest.TestCase):
    def write_sidecar(self, directory, rows):
        path = Path(directory) / "tracking.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["Frame", "RallyID", "Hit"])
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_pads_rally_and_counts_hits(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [
                {"Frame": frame, "RallyID": 1, "Hit": int(frame in (12, 18))}
                for frame in range(10, 21)
            ]
            path = self.write_sidecar(directory, rows)
            segments = read_rally_segments(
                path, 10.0, pre_roll_seconds=0.5, post_roll_seconds=0.7
            )
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].hit_count, 2)
        self.assertAlmostEqual(segments[0].start_seconds, 0.7)
        self.assertAlmostEqual(segments[0].end_seconds, 2.8)

    def test_adjacent_padding_is_made_non_overlapping(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = []
            for frame in range(10, 21):
                rows.append({"Frame": frame, "RallyID": 1, "Hit": 0})
            rows.append({"Frame": 21, "RallyID": 0, "Hit": 0})
            for frame in range(22, 31):
                rows.append({"Frame": frame, "RallyID": 2, "Hit": 0})
            path = self.write_sidecar(directory, rows)
            segments = read_rally_segments(
                path,
                10.0,
                pre_roll_seconds=0.8,
                post_roll_seconds=0.8,
                min_duration_seconds=0.1,
            )
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].end_seconds, segments[1].start_seconds)

    def test_zero_bridge_inside_rally_does_not_split_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_sidecar(directory, [
                {"Frame": 10, "RallyID": 4, "Hit": 1},
                {"Frame": 11, "RallyID": 4, "Hit": 0},
                {"Frame": 12, "RallyID": 0, "Hit": 0},
                {"Frame": 13, "RallyID": 0, "Hit": 0},
                {"Frame": 14, "RallyID": 4, "Hit": 0},
                {"Frame": 15, "RallyID": 4, "Hit": 1},
            ])
            segments = read_rally_segments(
                path,
                10.0,
                pre_roll_seconds=0.0,
                post_roll_seconds=0.0,
                min_duration_seconds=0.1,
            )
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].trajectory_start_frame, 10)
        self.assertEqual(segments[0].trajectory_end_frame, 15)
        self.assertEqual(segments[0].quality, "hit_anchored")

    def test_ignores_background_and_tiny_islands(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_sidecar(directory, [
                {"Frame": 0, "RallyID": 0, "Hit": 0},
                {"Frame": 1, "RallyID": 3, "Hit": 1},
                {"Frame": 2, "RallyID": 0, "Hit": 0},
            ])
            segments = read_rally_segments(
                path,
                25.0,
                pre_roll_seconds=0.0,
                post_roll_seconds=0.0,
                min_duration_seconds=1.0,
            )
        self.assertEqual(segments, [])


if __name__ == "__main__":
    unittest.main()
