from __future__ import annotations

from urllib.parse import quote
import requests

from src.config import load_vendor_risk_api_url


def get_vendor_risk(vendor_name: str, timeout_seconds: float = 3.0) -> dict:
    """Low-level API client. Decide yourself whether/how this becomes an agent tool."""
    base_url = load_vendor_risk_api_url()
    url = f"{base_url}/vendor-risk/{quote(vendor_name, safe='')}"
    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    return response.json()
