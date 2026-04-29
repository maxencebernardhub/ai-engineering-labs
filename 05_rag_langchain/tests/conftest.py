

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests requiring real embeddings + ChromaDB"
        " (deselect with '-m \"not integration\"')",
    )
