"""ASGI middleware: request ids and log lines, 500 responses, body size limit.

Architecture §5.9 and §10. Case IDs refer to docs/edge-case.md §4 and §8.
"""

import json
import sys
import uuid

from starlette.datastructures import MutableHeaders
from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.schemas import ServerErrorResponse
from app.observability import request_log

# A full request (location, 10 cuisines, 300 characters of free text) is under 2 KB.
MAX_BODY_BYTES = 16 * 1024


async def send_json(send: Send, status: int, body: dict) -> None:
    data = json.dumps(body).encode()
    headers = [(b"content-type", b"application/json"), (b"content-length", b"%d" % len(data))]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": data})


class RequestLogMiddleware:
    """Gives each request an id and writes its one log line; turns crashes into 500s.

    The id goes back in the X-Request-ID header, and in the body of a 500, so a user
    can quote it and the log line (with the traceback) can be found (EC-API-04).
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = uuid.uuid4().hex[:12]
        fields = {"request_id": request_id, "method": scope["method"], "path": scope["path"]}
        with request_log(**fields) as entry:

            async def send_with_id(message: Message) -> None:
                if message["type"] == "http.response.start":
                    entry["status"] = message["status"]
                    message.setdefault("headers", [])
                    MutableHeaders(scope=message).append("X-Request-ID", request_id)
                await send(message)

            try:
                await self.app(scope, receive, send_with_id)
            except Exception:
                entry["exc_info"] = sys.exc_info()
                if "status" in entry:  # the response had already started; nothing to send
                    raise
                # A generic message for the client; the traceback goes to the log only.
                body = ServerErrorResponse(detail="Internal server error", request_id=request_id)
                await send_json(send_with_id, 500, body.model_dump())


class BodySizeLimitMiddleware:
    """Rejects request bodies over `max_bytes` with 413, before reading them in full.

    A declared Content-Length is checked up front; a body sent without one (chunked) is
    counted as it arrives (EC-IN-54).
    """

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_BODY_BYTES):
        self.app = app
        self.max_bytes = max_bytes

    def _detail(self) -> str:
        return f"Request body too large; the limit is {self.max_bytes} bytes."

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        length = dict(scope["headers"]).get(b"content-length", b"")
        if length.isdigit() and int(length) > self.max_bytes:
            await send_json(send, 413, {"detail": self._detail()})
            return

        received = 0

        async def counted_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    # FastAPI re-raises HTTPException from body reading; Starlette turns
                    # it into a 413 response.
                    raise HTTPException(status_code=413, detail=self._detail())
            return message

        await self.app(scope, counted_receive, send)
