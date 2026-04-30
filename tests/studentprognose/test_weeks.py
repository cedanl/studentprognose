import pandas as pd

from studentprognose.utils.weeks import detect_last_available


class TestDetectLastAvailable:
    def test_detect_last_available_basic(self):
        data = pd.DataFrame(
            {
                "Collegejaar": [2023, 2023, 2023],
                "Weeknummer": [10, 20, 30],
            }
        )
        jaar, week = detect_last_available(data)
        assert jaar == 2023
        assert week == 30

    def test_detect_last_available_multiple_years(self):
        data = pd.DataFrame(
            {
                "Collegejaar": [2022, 2022, 2023, 2023, 2023],
                "Weeknummer": [10, 38, 5, 15, 25],
            }
        )
        jaar, week = detect_last_available(data)
        # Highest year is 2023, highest week in 2023 is 25
        assert jaar == 2023
        assert week == 25

    def test_detect_last_available_highest_week_in_highest_year(self):
        """Week from a lower year must not bleed into the result."""
        data = pd.DataFrame(
            {
                "Collegejaar": [2021, 2022, 2022],
                "Weeknummer": [52, 10, 18],
            }
        )
        jaar, week = detect_last_available(data)
        assert jaar == 2022
        assert week == 18

    def test_detect_last_available_split_academic_year(self):
        """Weken 39-52 (herfst) komen vóór weken 1-20 (lente) in het academisch jaar.

        Numeriek is week 52 hoger dan week 20, maar in het academisch jaar
        is week 20 recenter. De functie moet week 20 teruggeven.
        """
        weeks_fall = list(range(39, 53))   # 39 t/m 52
        weeks_spring = list(range(1, 21))  # 1 t/m 20
        all_weeks = weeks_fall + weeks_spring
        data = pd.DataFrame(
            {
                "Collegejaar": [2025] * len(all_weeks),
                "Weeknummer": all_weeks,
            }
        )
        jaar, week = detect_last_available(data)
        assert jaar == 2025
        assert week == 20  # niet 52

    def test_detect_last_available_no_weeknummer(self):
        """When Weeknummer column is absent, week should be None."""
        data = pd.DataFrame(
            {
                "Collegejaar": [2023, 2024],
            }
        )
        jaar, week = detect_last_available(data)
        assert jaar == 2024
        assert week is None
