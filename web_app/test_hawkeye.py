import unittest

from web_app.hawkeye import build_hawkeye_report


COURT_POINTS = [0, 0, 610, 0, 610, 1340, 0, 1340]


def _row(
    frame,
    *,
    hit=0,
    uncertain=False,
    hitter="",
    hitter_source="",
    x=None,
    y=None,
    confidence=0.95,
    hit_score=None,
    hit_source="",
    hitter_confidence=None,
    shot_end_frame=None,
):
    row = {
        "Frame": frame,
        "RallyID": 1,
        "Hit": hit,
        "HitScore": "" if hit_score is None else hit_score,
        "HitBallObserved": int(bool(hit)),
        "HitUncertain": int(bool(uncertain)),
        "HitSource": hit_source,
        "HitHitter": hitter,
        "HitHitterConfidence": "" if hitter_confidence is None else hitter_confidence,
        "HitHitterSource": hitter_source,
        "ShotEndFrame": "" if shot_end_frame is None else shot_end_frame,
        "RawVisibility": 0,
        "RawX": "",
        "RawY": "",
        "RawSource": "missing",
        "RawConfidence": "",
    }
    if x is not None and y is not None:
        row.update({
            "RawVisibility": 1,
            "RawX": x,
            "RawY": y,
            "RawSource": "model",
            "RawConfidence": confidence,
        })
    return row


class HawkEyeReviewTests(unittest.TestCase):
    def _report(self, tail):
        rows = [_row(0, hit=1, hitter="near"), *tail]
        return build_hawkeye_report(rows, fps=25.0, court_points=COURT_POINTS)

    def test_terminal_inside_point_is_an_in_candidate(self):
        report = self._report([
            _row(4, x=265, y=225),
            _row(5, x=287, y=208),
            _row(6, x=300, y=201),
            _row(7, x=301, y=200),
            _row(8, x=300, y=201),
        ])
        self.assertEqual(report["status"], "ready")
        candidate = report["candidates"][0]
        self.assertEqual(candidate["decision"], "in")
        self.assertEqual(candidate["method"], "visible_rest")
        self.assertGreater(candidate["landing"]["x_m"], 2.5)
        self.assertLess(candidate["landing"]["x_m"], 3.5)

    def test_terminal_outside_point_is_an_out_candidate(self):
        report = self._report([
            _row(4, x=650, y=226),
            _row(5, x=675, y=209),
            _row(6, x=690, y=201),
            _row(7, x=691, y=200),
            _row(8, x=690, y=201),
        ])
        self.assertEqual(report["status"], "ready")
        candidate = report["candidates"][0]
        self.assertEqual(candidate["decision"], "out")
        self.assertGreater(candidate["line_distance_cm"], 40)

    def test_bottom_baseline_near_outside_point_reports_short_margin(self):
        """A point just beyond the baseline must not become a metre-scale gap.

        The synthetic calibration is one centimetre per source pixel.  The
        final measured points are eight centimetres beyond the y=13.40 m
        baseline while x remains inside the doubles court.  This exercises the
        rectangle-boundary distance (the perpendicular baseline distance),
        rather than a distance to an unrelated corner or the full court edge.
        """
        report = build_hawkeye_report(
            [
                _row(0, hit=1, hitter="far"),
                _row(4, x=300, y=1300),
                _row(5, x=300, y=1332),
                _row(6, x=301, y=1348),
                _row(7, x=301, y=1349),
                _row(8, x=300, y=1348),
            ],
            fps=25.0,
            court_points=COURT_POINTS,
        )

        self.assertEqual(report["status"], "ready")
        candidate = report["candidates"][0]
        self.assertEqual(candidate["decision"], "review")
        self.assertEqual(candidate["method"], "visible_rest")
        self.assertGreaterEqual(candidate["line_distance_cm"], 6.0)
        self.assertLessEqual(candidate["line_distance_cm"], 10.0)

    def test_late_terminal_proxy_cannot_overstate_baseline_margin(self):
        """A short late detector branch must not replace a plausible endpoint.

        This mirrors the failure seen in the field: a coherent-looking
        three-frame response after a detector gap projects about 1.7 m beyond
        the baseline, even though the preceding measured flight reaches the
        line.  The later branch is a terminal endpoint *proxy*, not evidence
        that the shuttle landed another 1.7 m away.  Keep the earlier endpoint
        (or decline the candidate), but never publish that inflated margin.
        """
        court_points = [642, 479, 1275, 477, 1579, 1006, 343, 1008]
        rows = [
            _row(856, hit=1, hitter="near"),
            # A plausible flight ending at the near baseline (y ~= 0.05 m).
            _row(880, x=1000, y=500),
            _row(881, x=1035, y=490),
            _row(882, x=1070, y=483),
            _row(883, x=1108, y=479),
            _row(884, x=1109, y=479),
            # A gap is followed by a stationary false response that maps to
            # y ~= -1.7 m.  Current code incorrectly selects this run.
            _row(900, x=1100, y=446),
            _row(901, x=1100, y=445),
            _row(902, x=1100, y=446),
        ]

        report = build_hawkeye_report(
            rows,
            fps=25.0,
            court_points=court_points,
        )

        self.assertIn(report["status"], {"ready", "insufficient_evidence"})
        if report["status"] == "insufficient_evidence":
            # A strict implementation may decline the whole terminal branch;
            # that is safer than publishing a fabricated metre-scale margin.
            self.assertEqual(report["candidates"], [])
            return
        self.assertTrue(report["candidates"])
        candidate = report["candidates"][0]
        # The exact fallback policy may choose review or a no-call, but it
        # must not expose the terminal proxy's 170+ cm extrapolation.
        self.assertLess(candidate["line_distance_cm"], 50.0)
        self.assertGreater(candidate["landing"]["y_m"], -0.5)
        self.assertLessEqual(candidate["frame"], 884)

    def test_late_short_proxy_inside_court_does_not_replace_prior_endpoint(self):
        """A post-gap player lock is unsafe even when its projection is in-court."""
        rows = [
            _row(0, hit=1, hitter="near"),
            _row(4, x=265, y=225),
            _row(5, x=287, y=208),
            _row(6, x=300, y=201),
            _row(7, x=301, y=200),
            _row(8, x=300, y=201),
            # This branch is continuous but starts after a detector gap.  It
            # projects well inside the court, so a distance-only gate would
            # incorrectly select it as the terminal endpoint.
            _row(20, x=300, y=400),
            _row(21, x=300, y=400),
            _row(22, x=300, y=400),
        ]
        report = build_hawkeye_report(rows, fps=25.0, court_points=COURT_POINTS)
        self.assertEqual(report["status"], "ready")
        candidate = report["candidates"][0]
        self.assertEqual(candidate["frame"], 8)
        self.assertIn("discontinuous_terminal_track", [
            item["code"] for item in report["diagnostics"]["reason_counts"]
        ])

    def test_unsettled_terminal_tail_is_not_a_landing(self):
        """A moving endpoint proxy must not become a pseudo landing."""
        report = self._report([
            _row(4, x=300, y=500),
            _row(5, x=300, y=520),
            _row(6, x=300, y=540),
        ])

        self.assertEqual(report["status"], "insufficient_evidence")
        self.assertEqual(report["candidates"], [])
        self.assertIn("no_landing_evidence", {
            item["code"] for item in report["diagnostics"]["reason_counts"]
        })

    def test_far_hitter_target_half_is_preserved(self):
        """The far-side terminal run should use the opposite court half."""
        report = build_hawkeye_report(
            [
                _row(0, hit=1, hitter="far"),
                _row(4, x=300, y=900),
                _row(5, x=300, y=960),
                _row(6, x=301, y=1001),
                _row(7, x=301, y=1000),
                _row(8, x=300, y=1001),
            ],
            fps=25.0,
            court_points=COURT_POINTS,
        )

        self.assertEqual(report["status"], "ready")
        candidate = report["candidates"][0]
        self.assertEqual(candidate["hitter"], "far")
        self.assertEqual(candidate["decision"], "in")

    def test_sparse_point_near_line_is_not_a_landing(self):
        report = self._report([
            _row(6, x=607, y=205),
            _row(7, x=610, y=202),
            _row(8, x=612, y=200),
        ])
        self.assertEqual(report["status"], "insufficient_evidence")
        self.assertEqual(report["candidates"], [])

    def test_sparse_tail_is_suppressed_instead_of_guessing(self):
        report = self._report([
            _row(6, x=300, y=205),
            _row(8, x=302, y=200),
        ])
        self.assertEqual(report["status"], "insufficient_evidence")
        self.assertEqual(report["candidates"], [])

    def test_long_detector_gap_cannot_be_bridged_into_a_landing(self):
        report = self._report([
            _row(6, x=295, y=205),
            _row(9, x=300, y=202),
            _row(12, x=301, y=200),
        ])
        self.assertEqual(report["status"], "insufficient_evidence")

    def test_late_isolated_response_does_not_erase_prior_coherent_endpoint(self):
        report = self._report([
            _row(4, x=265, y=225),
            _row(5, x=287, y=208),
            _row(6, x=300, y=201),
            _row(7, x=301, y=200),
            _row(8, x=300, y=201),
            _row(25, x=350, y=210),
        ])
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["candidates"][0]["frame"], 8)

    def test_uncertain_late_hit_does_not_replace_last_confirmed_hit(self):
        rows = [
            _row(0, hit=1, hitter="near"),
            _row(4, x=265, y=225),
            _row(5, x=287, y=208),
            _row(6, x=300, y=201),
            _row(7, x=301, y=200),
            _row(8, x=300, y=201),
            # A HIT? marker is still a measured observation, but it must not
            # become the anchor that erases the preceding terminal flight.
            _row(25, hit=1, uncertain=True, hitter="far", x=350, y=210),
        ]

        report = build_hawkeye_report(
            rows,
            fps=25.0,
            court_points=COURT_POINTS,
        )

        self.assertEqual(report["status"], "ready")
        candidate = report["candidates"][0]
        self.assertEqual(candidate["hit_frame"], 0)
        self.assertEqual(candidate["frame"], 8)
        self.assertEqual(candidate["hitter"], "near")
        self.assertEqual(report["diagnostics"]["ignored_unconfirmed_hits"], 1)

    def test_shot_end_frame_blocks_landing_after_next_contact(self):
        rows = [
            _row(0, hit=1, hitter="near", shot_end_frame=25),
            _row(4, x=265, y=225),
            _row(5, x=287, y=208),
            _row(6, x=300, y=201),
            _row(7, x=301, y=200),
            _row(8, x=300, y=201),
        ]
        report = build_hawkeye_report(
            rows, fps=25.0, court_points=COURT_POINTS
        )
        self.assertEqual(report["status"], "insufficient_evidence")
        self.assertEqual(report["candidates"], [])
        self.assertEqual(
            report["diagnostics"]["reason_counts"],
            [{"code": "shot_continues_to_next_contact", "count": 1}],
        )

    def test_strong_uncertain_late_hit_blocks_landing(self):
        rows = [
            _row(0, hit=1, hitter="near"),
            _row(4, x=265, y=225),
            _row(5, x=287, y=208),
            _row(6, x=300, y=201),
            _row(7, x=301, y=200),
            _row(8, x=300, y=201),
            _row(
                25,
                hit=1,
                uncertain=True,
                hitter="far",
                hit_score=0.70,
                hitter_confidence=0.90,
                hitter_source="pose_contact",
            ),
        ]
        report = build_hawkeye_report(
            rows, fps=25.0, court_points=COURT_POINTS
        )
        self.assertEqual(report["status"], "insufficient_evidence")
        self.assertEqual(report["candidates"], [])
        self.assertEqual(
            report["diagnostics"]["reason_counts"],
            [{"code": "later_hit_unconfirmed", "count": 1}],
        )

    def test_same_run_body_lock_is_not_a_landing(self):
        # A detector can jump from a real flight onto a player's clothing and
        # then remain nearly stationary.  The approach jump is too large to
        # be a physical speed drop, so no court point is published.
        report = self._report([
            _row(4, x=260, y=260),
            _row(5, x=282, y=238),
            _row(6, x=302, y=218),
            _row(7, x=322, y=198),
            _row(8, x=900, y=500),
            _row(9, x=900, y=500),
            _row(10, x=901, y=500),
        ])
        self.assertEqual(report["status"], "insufficient_evidence")
        self.assertEqual(report["candidates"], [])
        self.assertIn("no_landing_evidence", {
            item["code"] for item in report["diagnostics"]["reason_counts"]
        })

    def test_five_point_late_branch_after_gap_cannot_replace_rest(self):
        report = self._report([
            _row(4, x=265, y=225),
            _row(5, x=287, y=208),
            _row(6, x=300, y=201),
            _row(7, x=301, y=200),
            _row(8, x=300, y=201),
            _row(20, x=420, y=420),
            _row(21, x=420, y=420),
            _row(22, x=420, y=420),
            _row(23, x=420, y=420),
            _row(24, x=420, y=420),
        ])
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["candidates"][0]["frame"], 8)
        self.assertIn("discontinuous_terminal_track", {
            item["code"] for item in report["diagnostics"]["reason_counts"]
        })

    def test_implausible_late_run_falls_back_to_prior_terminal_run(self):
        # This trapezoid mirrors a broadcast view.  The final three measured
        # points form a coherent detector branch, but their homography result
        # is almost seven metres beyond the far baseline.  That impossible
        # branch must not erase the earlier, plausible terminal observation.
        court_points = [642, 483, 1273, 479, 1579, 1002, 349, 1008]
        rows = [
            _row(0, hit=1, hitter="near"),
            _row(4, x=850, y=620),
            _row(5, x=880, y=605),
            _row(6, x=900, y=600),
            _row(7, x=901, y=600),
            _row(8, x=900, y=601),
            _row(25, x=1323, y=367),
            _row(26, x=1321, y=367),
            _row(27, x=1319, y=368),
        ]

        report = build_hawkeye_report(
            rows,
            fps=25.0,
            court_points=court_points,
        )

        self.assertEqual(report["status"], "ready")
        candidate = report["candidates"][0]
        self.assertEqual(candidate["hit_frame"], 0)
        self.assertEqual(candidate["frame"], 8)
        self.assertEqual(candidate["decision"], "review")
        self.assertLess(candidate["confidence"], 82)
        self.assertEqual(report["diagnostics"]["fallback_candidate_count"], 1)
        self.assertEqual(
            report["diagnostics"]["reason_counts"],
            [{"code": "discontinuous_terminal_track", "count": 1}],
        )

    def test_own_half_terminal_track_is_not_a_line_call(self):
        report = self._report([
            _row(6, x=300, y=1002),
            _row(7, x=302, y=1000),
            _row(8, x=304, y=998),
        ])
        self.assertEqual(report["status"], "insufficient_evidence")

    def test_unknown_hitter_never_receives_a_hard_line_call(self):
        rows = [
            _row(0, hit=1, hitter=""),
            _row(4, x=265, y=225),
            _row(5, x=287, y=208),
            _row(6, x=300, y=201),
            _row(7, x=301, y=200),
            _row(8, x=300, y=201),
        ]
        report = build_hawkeye_report(rows, fps=25.0, court_points=COURT_POINTS)
        self.assertEqual(report["status"], "ready")
        candidate = report["candidates"][0]
        self.assertEqual(candidate["method"], "visible_rest")
        self.assertEqual(candidate["decision"], "review")
        self.assertLess(candidate["confidence"], 82)

    def test_rally_alternation_hitter_is_review_only_context(self):
        # The terminal run lands in the apparent hitter's own half.  A hard
        # side label would suppress it before review; sequence-inferred sides
        # must instead keep the video evidence and force a no-call.
        tail = [
            _row(4, x=265, y=1025),
            _row(5, x=287, y=1010),
            _row(6, x=300, y=1001),
            _row(7, x=301, y=1000),
            _row(8, x=300, y=1001),
        ]
        for source in ("rally_alternation_inferred", "rally_alternation_corrected"):
            with self.subTest(source=source):
                report = build_hawkeye_report(
                    [_row(0, hit=1, hitter="near", hitter_source=source), *tail],
                    fps=25.0,
                    court_points=COURT_POINTS,
                )
                self.assertEqual(report["status"], "ready")
                candidate = report["candidates"][0]
                self.assertEqual(candidate["hitter"], "unknown")
                self.assertTrue(candidate["hitter_review_only"])
                self.assertEqual(candidate["decision"], "review")
                self.assertLess(candidate["confidence"], 82)
                self.assertIn("统一降级为待复核", report["notice"])

    def test_cross_cut_mode_forces_manual_review(self):
        rows = [
            _row(0, hit=1, hitter="near"),
            _row(4, x=265, y=225),
            _row(5, x=287, y=208),
            _row(6, x=300, y=201),
            _row(7, x=301, y=200),
            _row(8, x=300, y=201),
        ]
        report = build_hawkeye_report(
            rows,
            fps=25.0,
            court_points=COURT_POINTS,
            force_review=True,
        )
        self.assertEqual(report["candidates"][0]["decision"], "review")


if __name__ == "__main__":
    unittest.main()
