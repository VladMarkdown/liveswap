"""IP camera capture using the common HTTP /video endpoint."""
from urllib.parse import urlsplit, urlunsplit

from .capture import CaptureThread


class IPCameraCapture(CaptureThread):
    def __init__(self, url: str, **kwargs):
        if "://" not in url:
            url = "http://" + url
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https", "rtsp") or not parts.netloc:
            raise ValueError("Expected an HTTP, HTTPS, or RTSP camera URL")
        if parts.scheme in ("http", "https") and parts.path in ("", "/"):
            url = urlunsplit(parts._replace(path="/video"))
        super().__init__(url, **kwargs)
