from known_models import MODEL_OVERRIDES


class TestModelOverridesStructure:
    def test_all_values_are_3_tuples(self):
        for key, value in MODEL_OVERRIDES.items():
            assert isinstance(value, tuple) and len(value) == 3, (
                f"Override '{key}' should be a 3-tuple, got {value!r}"
            )

    def test_architecture_is_valid(self):
        for key, (_, _, arch) in MODEL_OVERRIDES.items():
            assert arch in ("dense", "moe"), (
                f"Override '{key}' has invalid architecture '{arch}'"
            )

    def test_total_params_positive(self):
        for key, (total, _, _) in MODEL_OVERRIDES.items():
            assert total > 0, f"Override '{key}' has non-positive total_params: {total}"

    def test_active_params_positive(self):
        for key, (_, active, _) in MODEL_OVERRIDES.items():
            assert active > 0, f"Override '{key}' has non-positive active_params: {active}"

    def test_active_lte_total(self):
        for key, (total, active, _) in MODEL_OVERRIDES.items():
            assert active <= total, (
                f"Override '{key}' has active ({active}) > total ({total})"
            )

    def test_moe_active_less_than_total(self):
        for key, (total, active, arch) in MODEL_OVERRIDES.items():
            if arch == "moe":
                assert active < total, (
                    f"MoE override '{key}' should have active < total, "
                    f"got active={active}, total={total}"
                )

    def test_dense_active_equals_total(self):
        for key, (total, active, arch) in MODEL_OVERRIDES.items():
            if arch == "dense":
                assert active == total, (
                    f"Dense override '{key}' should have active == total, "
                    f"got active={active}, total={total}"
                )

    def test_intentional_case_duplicates_only(self):
        """Check that lowercase key collisions are intentional (like Phi-4/phi-4)."""
        lower_keys = {}
        for key in MODEL_OVERRIDES:
            lk = key.lower()
            if lk in lower_keys:
                # Both entries should have identical values
                assert MODEL_OVERRIDES[key] == MODEL_OVERRIDES[lower_keys[lk]], (
                    f"Case-duplicate keys '{lower_keys[lk]}' and '{key}' "
                    f"have different values"
                )
            lower_keys[lk] = key
