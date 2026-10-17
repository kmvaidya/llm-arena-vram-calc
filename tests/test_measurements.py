import statistics

from measurements import MEASUREMENTS
from vram_calculator import PRECISION_BYTES, decode_tokens_per_s


def _errors():
    return [
        decode_tokens_per_s(m.active_params_b, m.architecture, m.precision,
                            m.bandwidth_gbs, m.platform, m.gpu_count) / m.tokens_per_s - 1
        for m in MEASUREMENTS
    ]


class TestMeasurements:
    def test_entries_are_well_formed(self):
        for m in MEASUREMENTS:
            assert m.architecture in ("dense", "moe")
            assert m.precision in PRECISION_BYTES
            assert m.platform in ("gpu", "apple")
            assert m.source.startswith("https://")
            assert m.active_params_b > 0 and m.tokens_per_s > 0

    def test_covers_every_kind_of_hardware(self):
        assert any(m.platform == "apple" for m in MEASUREMENTS)
        assert any(m.gpu_count == 8 for m in MEASUREMENTS)
        assert any(m.architecture == "moe" and m.gpu_count == 1 for m in MEASUREMENTS)


class TestSpeedFormulaAccuracy:
    """Guards the speed constants: a change that fits the measurements worse fails here."""

    def test_median_error_within_15_percent(self):
        assert statistics.median(abs(e) for e in _errors()) <= 0.15

    def test_most_estimates_within_20_percent(self):
        errors = _errors()
        assert sum(abs(e) <= 0.2 for e in errors) / len(errors) >= 0.65

    def test_no_systematic_bias(self):
        # Over- and underestimates roughly balance
        assert abs(statistics.median(_errors())) <= 0.1
