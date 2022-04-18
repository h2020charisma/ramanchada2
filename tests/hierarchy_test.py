#!/usr/bin/env python3

import pytest


def test_hierarchy():
    import ramanchada2
    assert hasattr(ramanchada2, 'spectrum')
    assert hasattr(ramanchada2, 'theoretical_lines')
    with pytest.raises(DeprecationWarning):
        import ramanchada2.spectral_components
