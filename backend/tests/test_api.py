import unittest
from datetime import datetime, timezone
from unittest.mock import Mock

from fastapi.testclient import TestClient

from main import (
    DAILY_REQUEST_LIMIT,
    _daily_request_counts,
    _daily_request_counts_lock,
    app,
    consume_daily_request,
    get_service,
)


class ApiTests(unittest.TestCase):
    def setUp(self):
        with _daily_request_counts_lock:
            _daily_request_counts.clear()
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()

    def test_retrieve_endpoint_is_not_exposed(self):
        response = self.client.post("/api/retrieve", json={"question": "BGP?"})

        self.assertEqual(response.status_code, 404)

    def test_daily_limit_blocks_before_retrieval_and_returns_retry_after(self):
        service = Mock()
        service.retrieve.return_value = []
        app.dependency_overrides[get_service] = lambda: service
        headers = {"X-Forwarded-For": "198.51.100.10"}

        responses = [
            self.client.post("/api/ask", json={"question": "BGP?"}, headers=headers)
            for _ in range(DAILY_REQUEST_LIMIT)
        ]
        limited = self.client.post(
            "/api/ask", json={"question": "BGP?"}, headers=headers
        )

        self.assertTrue(all(response.status_code == 200 for response in responses))
        self.assertEqual(limited.status_code, 429)
        self.assertIn("Retry-After", limited.headers)
        self.assertEqual(service.retrieve.call_count, DAILY_REQUEST_LIMIT)

    def test_daily_limit_is_independent_per_ip_and_resets_on_next_utc_day(self):
        now = datetime(2026, 11, 22, 23, 59, tzinfo=timezone.utc)
        for _ in range(DAILY_REQUEST_LIMIT):
            self.assertIsNone(consume_daily_request("198.51.100.11", now))

        self.assertGreater(consume_daily_request("198.51.100.11", now), 0)
        self.assertIsNone(consume_daily_request("198.51.100.12", now))
        self.assertIsNone(
            consume_daily_request(
                "198.51.100.11", datetime(2026, 11, 23, 0, 0, tzinfo=timezone.utc)
            )
        )


if __name__ == "__main__":
    unittest.main()
