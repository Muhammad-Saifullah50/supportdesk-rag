import unittest
from unittest.mock import Mock, patch

from providers import ProviderError, post_with_retries


class ProviderRetryTests(unittest.TestCase):
    def test_retries_transient_status_and_returns_success(self):
        failed = Mock(status_code=429, headers={})
        retry_success = Mock(status_code=200, headers={})
        post = Mock(side_effect=[failed, retry_success])

        with patch("providers.time.sleep"):
            response = post_with_retries(
                post,
                provider="test-provider",
                url="/test",
                json={"question": "test"},
            )

        self.assertIs(response, retry_success)
        self.assertEqual(post.call_count, 2)

    def test_exposes_provider_status_after_retry_budget(self):
        response = Mock(status_code=503, headers={"Retry-After": "2"})
        post = Mock(return_value=response)

        with patch("providers.time.sleep"):
            with self.assertRaises(ProviderError) as raised:
                post_with_retries(
                    post,
                    provider="test-provider",
                    url="/test",
                    json={},
                )

        self.assertEqual(post.call_count, 3)
        self.assertEqual(raised.exception.provider, "test-provider")
        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(raised.exception.retry_after, "2")


if __name__ == "__main__":
    unittest.main()
