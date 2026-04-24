"""
Shared client/server connection configuration helpers.
"""

import copy
import json
import os
from typing import Any, Dict, Optional


DEFAULT_CONFIG: Dict[str, Any] = {
    "server": {
        "server_url": "http://localhost:6006",
        "host": "localhost",
        "port": 6006,
    },
    "ssh": {
        "enabled": True,
        "ssh_host": "",
        "ssh_port": 22,
        "ssh_username": "root",
        "ssh_password": "",
        "remote_host": "127.0.0.1",
        "remote_port": 6006,
        "local_port": 6006,
    },
    "timeouts": {
        "connect": 20,
        "request": 30,
        "status": 10,
        "download": 300,
        "task": 3600,
    },
    "retry": {
        "max_reconnect_attempts": 10,
        "initial_delay": 1.0,
        "max_delay": 30.0,
        "max_status_errors": 5,
        "http_max_retries": 3,
    },
    "progress": {
        "poll_interval": 5,
        "heartbeat_interval": 25.0,
    },
}


def get_default_config_path() -> str:
    """Return the default client/server config path."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_client_server.json")


def get_example_config_path() -> str:
    """Return the tracked example client/server config path."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_client_server.example.json")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_client_server_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load client config and merge it over safe defaults."""
    config = copy.deepcopy(DEFAULT_CONFIG)
    path = config_path or get_default_config_path()

    if not os.path.exists(path):
        return config

    with open(path, "r", encoding="utf-8-sig") as f:
        file_config = json.load(f)

    if not isinstance(file_config, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")

    return _deep_merge(config, file_config)


def build_server_url(config: Dict[str, Any]) -> str:
    """Build the HTTP server URL from config."""
    server_config = config.get("server", {})
    server_url = server_config.get("server_url")
    if server_url:
        return str(server_url).rstrip("/")

    host = server_config.get("host", "localhost")
    port = int(server_config.get("port", 6006))
    return f"http://{host}:{port}"
