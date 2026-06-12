import pandas as pd

from studentprognose.utils.weeks import (
    detect_last_available,
    academic_start_week,
    get_all_weeks_ordered,
    compute_pred_len,
    get_weeks_list,
    week_sort_key,
)
from studentprognose.config import get_final_academic_week


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


class TestConfigurableFinalWeek:
    """De academische week-window is config-gedreven (legacy 38, UvA 36) — issue #231."""

    def test_academic_start_week(self):
        assert academic_start_week(38) == 39   # legacy: reset week 39
        assert academic_start_week(36) == 37   # UvA: reset week 37
        assert academic_start_week(52) == 1    # wrap

    def test_get_all_weeks_ordered_legacy_38(self):
        weeks = get_all_weeks_ordered(38)
        assert weeks[0] == "39"
        assert weeks[-1] == "38"
        assert len(weeks) == 52

    def test_get_all_weeks_ordered_uva_36(self):
        weeks = get_all_weeks_ordered(36)
        assert weeks[:3] == ["37", "38", "39"]   # reset-weken aan het begin
        assert weeks[-1] == "36"                  # seizoen eindigt op 36
        assert len(weeks) == 52
        assert "53" not in weeks                  # 52-weeks model

    def test_compute_pred_len_respects_final_week(self):
        assert compute_pred_len(12, 38) == 26    # 38 - 12
        assert compute_pred_len(12, 36) == 24    # 36 - 12
        assert compute_pred_len(40, 38) == 50    # wrap: 38 + 52 - 40

    def test_get_weeks_list_is_full_year_for_both(self):
        assert len(get_weeks_list(38, 38)) == 52
        assert len(get_weeks_list(36, 36)) == 52

    def test_week_sort_key_orders_each_season(self):
        assert week_sort_key(39, 38) == 0    # legacy start eerst
        assert week_sort_key(38, 38) == 52   # legacy eind laatst (52: ruimte voor wk53)
        assert week_sort_key(37, 36) == 0    # UvA start eerst
        assert week_sort_key(36, 36) == 52   # UvA eind laatst

    def test_week_sort_key_iso_week_53_between_52_and_1(self):
        # Lange ISO-jaren (2021/2026) hebben week 53; die hoort tussen 52 en 1,
        # zonder botsing met week 1 (regressie op de #231-review-finding).
        assert week_sort_key(52, 36) < week_sort_key(53, 36) < week_sort_key(1, 36)
        assert week_sort_key(53, 36) != week_sort_key(1, 36)
        assert week_sort_key(52, 38) < week_sort_key(53, 38) < week_sort_key(1, 38)

    def test_get_final_academic_week_default_and_override(self):
        assert get_final_academic_week({}) == 38
        assert get_final_academic_week({"model_config": {"final_academic_week": 36}}) == 36
