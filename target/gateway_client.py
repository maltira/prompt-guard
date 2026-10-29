"""клиент шлюза для мишени в режиме gateway"""
import httpx


class GatewayError(RuntimeError):
    pass


class GatewayClient:
    def __init__(self, url: str, timeout_s: float = 60.0):
        self.url, self.timeout_s = url.rstrip("/"), timeout_s  # судья отвечает секунды, поэтому запас по времени

    def check(self, subject: str, body: str, detector: str | None = None) -> dict:
        payload = {"subject": subject, "body": body}
        if detector:
            payload["detector"] = detector
        try:
            r = httpx.post(f"{self.url}/check", json=payload, timeout=self.timeout_s)
        except httpx.HTTPError as e:
            raise GatewayError(f"шлюз недоступен: {e}") from e
        if r.status_code != 200:
            raise GatewayError(f"шлюз ответил {r.status_code}: {r.text[:200]}")
        return r.json()
