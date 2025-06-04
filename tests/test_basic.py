"""Basic tests for the WronAI package."""

import pytest

from wronai import __version__


def test_version():
    """Test that the package has a version string."""
    assert __version__ is not None
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_imports():
    """Test that the package can be imported."""
    # Test that we can import the main module
    from wronai import cli_new  # noqa: F401

    assert True  # If we get here, the import worked


if __name__ == "__main__":
    pytest.main([__file__])
