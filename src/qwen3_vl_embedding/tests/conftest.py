from pytest import Config


def pytest_configure(config: Config) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "asyncio: mark test as async")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line(
        "markers", "requires_api: mark test as requiring API server"
    )
