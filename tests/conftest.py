import pytest

import app as app_module


@pytest.fixture(autouse=True)
def reset_rate_limiters():
    # Limiters are module-level, so without this one test's requests count against the next
    for limiter in (
        app_module.query_limiter_per_ip,
        app_module.query_limiter_global,
        app_module.upload_limiter_per_ip,
    ):
        limiter.reset()
    yield
