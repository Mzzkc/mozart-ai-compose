"""Tests for Mozart Dashboard authentication module."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from mozart.dashboard.auth import (
    AuthConfig,
    AuthMiddleware,
    AuthMode,
    generate_api_key,
    hash_api_key,
    is_localhost,
    require_api_key,
    verify_api_key,
)


class TestAuthConfig:
    """Test AuthConfig configuration class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AuthConfig()

        assert config.mode == AuthMode.LOCALHOST_ONLY
        assert config.api_keys == []
        assert config.localhost_bypass is True
        assert "/health" in config.excluded_paths
        assert "/docs" in config.excluded_paths
        assert config.header_name == "X-API-Key"

    def test_from_env_defaults(self):
        """Test config from environment with defaults."""
        with patch.dict("os.environ", {}, clear=True):
            config = AuthConfig.from_env()

            assert config.mode == AuthMode.LOCALHOST_ONLY
            assert config.api_keys == []
            assert config.localhost_bypass is True

    def test_from_env_api_key_mode(self):
        """Test config from environment with API key mode."""
        env = {
            "MOZART_AUTH_MODE": "api_key",
            "MOZART_API_KEYS": "key1,key2,key3",
            "MOZART_LOCALHOST_BYPASS": "false",
        }
        with patch.dict("os.environ", env, clear=True):
            config = AuthConfig.from_env()

            assert config.mode == AuthMode.API_KEY
            assert config.api_keys == [hash_api_key("key1"), hash_api_key("key2"), hash_api_key("key3")]
            assert config.localhost_bypass is False

    def test_from_env_disabled_mode(self):
        """Test config with disabled authentication."""
        env = {"MOZART_AUTH_MODE": "disabled"}
        with patch.dict("os.environ", env, clear=True):
            config = AuthConfig.from_env()

            assert config.mode == AuthMode.DISABLED

    def test_from_env_empty_api_keys(self):
        """Test config with empty API keys string."""
        env = {"MOZART_API_KEYS": ""}
        with patch.dict("os.environ", env, clear=True):
            config = AuthConfig.from_env()

            assert config.api_keys == []


class TestApiKeyFunctions:
    """Test API key utility functions."""

    def test_hash_api_key(self):
        """Test API key hashing produces consistent results."""
        key = "test-api-key-123"
        hash1 = hash_api_key(key)
        hash2 = hash_api_key(key)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length
        assert hash1 != key  # Should not equal original

    def test_hash_api_key_different_inputs(self):
        """Test different keys produce different hashes."""
        hash1 = hash_api_key("key1")
        hash2 = hash_api_key("key2")

        assert hash1 != hash2

    def test_verify_api_key_valid(self):
        """Test verifying a valid API key."""
        key = "valid-test-key"
        stored_hash = hash_api_key(key)

        assert verify_api_key(key, [stored_hash]) is True

    def test_verify_api_key_invalid(self):
        """Test verifying an invalid API key."""
        valid_key = "valid-test-key"
        stored_hash = hash_api_key(valid_key)

        assert verify_api_key("wrong-key", [stored_hash]) is False

    def test_verify_api_key_multiple_valid(self):
        """Test verifying against multiple valid keys."""
        keys = ["key1", "key2", "key3"]
        hashes = [hash_api_key(k) for k in keys]

        assert verify_api_key("key2", hashes) is True
        assert verify_api_key("invalid", hashes) is False

    def test_verify_api_key_empty_list(self):
        """Test verifying against empty key list."""
        assert verify_api_key("any-key", []) is False

    def test_generate_api_key_unique(self):
        """Test generated API keys are unique."""
        keys = [generate_api_key() for _ in range(100)]

        assert len(set(keys)) == 100  # All unique

    def test_generate_api_key_length(self):
        """Test generated API keys have appropriate length."""
        key = generate_api_key()

        # URL-safe base64 encoding of 32 bytes
        assert len(key) >= 40


class TestIsLocalhost:
    """Test localhost detection."""

    def test_is_localhost_ipv4(self):
        """Test localhost detection for IPv4."""
        request = MagicMock()
        request.client.host = "127.0.0.1"

        assert is_localhost(request) is True

    def test_is_localhost_ipv6(self):
        """Test localhost detection for IPv6."""
        request = MagicMock()
        request.client.host = "::1"

        assert is_localhost(request) is True

    def test_is_localhost_hostname(self):
        """Test localhost detection for hostname."""
        request = MagicMock()
        request.client.host = "localhost"

        assert is_localhost(request) is True

    def test_is_not_localhost(self):
        """Test non-localhost detection."""
        request = MagicMock()
        request.client.host = "192.168.1.100"

        assert is_localhost(request) is False

    def test_is_localhost_no_client(self):
        """Test handling when client is None."""
        request = MagicMock()
        request.client = None

        assert is_localhost(request) is False


class TestAuthMiddleware:
    """Test authentication middleware."""

    @pytest.fixture
    def create_app(self):
        """Factory fixture to create test apps with different configs."""

        def _create_app(config: AuthConfig | None = None) -> FastAPI:
            app = FastAPI()

            @app.get("/")
            async def root():
                return {"status": "ok"}

            @app.get("/health")
            async def health():
                return {"healthy": True}

            @app.get("/api/data")
            async def api_data():
                return {"data": "sensitive"}

            app.add_middleware(AuthMiddleware, config=config)
            return app

        return _create_app

    def test_disabled_mode_allows_all(self, create_app):
        """Test disabled auth mode allows all requests."""
        config = AuthConfig(mode=AuthMode.DISABLED)
        app = create_app(config)
        client = TestClient(app)

        response = client.get("/api/data")
        assert response.status_code == 200

    def test_localhost_only_allows_localhost(self, create_app):
        """Test localhost_only mode allows localhost."""
        config = AuthConfig(mode=AuthMode.LOCALHOST_ONLY)
        app = create_app(config)
        client = TestClient(app)

        # TestClient uses localhost by default
        response = client.get("/api/data")
        assert response.status_code == 200

    def test_excluded_paths_bypass_auth(self, create_app):
        """Test excluded paths bypass authentication."""
        config = AuthConfig(
            mode=AuthMode.API_KEY,
            api_keys=[hash_api_key("secret")],
            localhost_bypass=False,
        )
        app = create_app(config)
        client = TestClient(app)

        # /health is excluded by default
        response = client.get("/health")
        assert response.status_code == 200

    def test_api_key_required_without_bypass(self, create_app):
        """Test API key required when localhost bypass disabled."""
        config = AuthConfig(
            mode=AuthMode.API_KEY,
            api_keys=[hash_api_key("valid-key")],
            localhost_bypass=False,
        )
        app = create_app(config)
        client = TestClient(app)

        # Request without API key
        response = client.get("/api/data")
        assert response.status_code == 401
        assert "API key required" in response.json()["detail"]

    def test_api_key_valid(self, create_app):
        """Test valid API key allows access."""
        config = AuthConfig(
            mode=AuthMode.API_KEY,
            api_keys=[hash_api_key("valid-key")],
            localhost_bypass=False,
        )
        app = create_app(config)
        client = TestClient(app)

        response = client.get("/api/data", headers={"X-API-Key": "valid-key"})
        assert response.status_code == 200

    def test_api_key_invalid(self, create_app):
        """Test invalid API key is rejected."""
        config = AuthConfig(
            mode=AuthMode.API_KEY,
            api_keys=[hash_api_key("valid-key")],
            localhost_bypass=False,
        )
        app = create_app(config)
        client = TestClient(app)

        response = client.get("/api/data", headers={"X-API-Key": "wrong-key"})
        assert response.status_code == 401
        assert "Invalid API key" in response.json()["detail"]

    def test_api_key_with_localhost_bypass(self, create_app):
        """Test API key mode with localhost bypass enabled."""
        config = AuthConfig(
            mode=AuthMode.API_KEY,
            api_keys=[hash_api_key("secret")],
            localhost_bypass=True,  # Default
        )
        app = create_app(config)
        client = TestClient(app)

        # Should work without API key from localhost
        response = client.get("/api/data")
        assert response.status_code == 200


class TestRequireApiKeyDependency:
    """Test the require_api_key FastAPI dependency."""

    @pytest.fixture
    def app_with_dependency(self):
        """Create app using dependency instead of middleware."""
        from fastapi import Depends

        app = FastAPI()

        @app.get("/public")
        async def public():
            return {"public": True}

        @app.get("/protected")
        async def protected(key: str = Depends(require_api_key)):
            return {"key": key}

        return app

    def test_dependency_allows_localhost(self, app_with_dependency):
        """Test dependency allows localhost without key."""
        with patch.dict("os.environ", {"MOZART_LOCALHOST_BYPASS": "true"}):
            client = TestClient(app_with_dependency)
            response = client.get("/protected")

            # Localhost bypass returns "localhost" as key
            assert response.status_code == 200
            assert response.json()["key"] == "localhost"

    def test_dependency_requires_key_no_bypass(self, app_with_dependency):
        """Test dependency requires key when bypass disabled."""
        env = {
            "MOZART_LOCALHOST_BYPASS": "false",
            "MOZART_API_KEYS": "valid-key",
        }
        with patch.dict("os.environ", env, clear=True):
            client = TestClient(app_with_dependency)

            # Without key
            response = client.get("/protected")
            assert response.status_code == 401


class TestAuthModeEnum:
    """Test AuthMode enumeration."""

    def test_mode_values(self):
        """Test all auth mode values exist."""
        assert AuthMode.DISABLED.value == "disabled"
        assert AuthMode.API_KEY.value == "api_key"
        assert AuthMode.LOCALHOST_ONLY.value == "localhost_only"

    def test_mode_from_string(self):
        """Test creating mode from string value."""
        assert AuthMode("disabled") == AuthMode.DISABLED
        assert AuthMode("api_key") == AuthMode.API_KEY
        assert AuthMode("localhost_only") == AuthMode.LOCALHOST_ONLY


class TestSecurityBestPractices:
    """Test security-related functionality."""

    def test_constant_time_comparison(self):
        """Test that key verification uses constant-time comparison."""
        # This is a basic test - true timing attack prevention requires
        # statistical analysis, but we verify hmac.compare_digest is used
        import inspect

        source = inspect.getsource(verify_api_key)
        assert "hmac.compare_digest" in source

    def test_api_key_not_logged(self):
        """Test that API keys are not exposed in errors."""
        config = AuthConfig(
            mode=AuthMode.API_KEY,
            api_keys=["super-secret-key"],
            localhost_bypass=False,
        )

        app = FastAPI()

        @app.get("/")
        async def root():
            return {"ok": True}

        app.add_middleware(AuthMiddleware, config=config)
        client = TestClient(app)

        response = client.get("/", headers={"X-API-Key": "wrong-key"})

        # Error message should not contain the actual keys
        assert "super-secret-key" not in response.text
        assert "wrong-key" not in response.text

    def test_hash_irreversibility(self):
        """Test that hashed keys cannot easily reveal original."""
        key = "my-secret-api-key"
        hashed = hash_api_key(key)

        # Hash should not contain key substring
        assert key not in hashed
        # Should be hexadecimal
        assert all(c in "0123456789abcdef" for c in hashed)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
