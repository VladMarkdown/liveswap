"""WebRTC output interface reserved for Phase 2."""


class WebRTCOutput:
    """TODO Phase 2: signaling, peer connections, and video track transport."""

    def start(self):
        raise NotImplementedError("WebRTC output is planned for Phase 2")

    def send_frame(self, frame):
        raise NotImplementedError("WebRTC output is planned for Phase 2")

    def close(self):
        pass
