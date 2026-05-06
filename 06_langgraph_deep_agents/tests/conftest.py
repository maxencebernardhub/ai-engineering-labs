import shutil
from pathlib import Path

import pytest


@pytest.fixture
def leads_file(tmp_path):
    source = Path(__file__).parent.parent / "data" / "leads.json"
    dest = tmp_path / "leads.json"
    shutil.copy(source, dest)
    return dest


@pytest.fixture
def drafts_dir(tmp_path):
    d = tmp_path / "drafts"
    d.mkdir()
    return d


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests requiring a real LLM"
        " (deselect with '-m \"not integration\"')",
    )
