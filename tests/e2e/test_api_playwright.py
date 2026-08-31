"""Browser smoke tests for the running Docker API."""

import pytest
from playwright.sync_api import Error as PlaywrightError


@pytest.mark.e2e
def test_swagger_docs_load(page, base_url):
    try:
        response = page.goto(f"{base_url}/docs")
    except PlaywrightError as exc:
        if "ERR_CONNECTION_REFUSED" in str(exc):
            pytest.skip(
                f"API is not running at {base_url}; start it with "
                "docker compose up --build -d"
            )
        raise

    assert response is not None
    assert response.ok
    assert "Swagger UI" in page.title()
    assert page.locator("#swagger-ui").is_visible()
