"""FastAPI middleware for request tracing, structured logging, and error boundaries."""

import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.core.errors import ClinGraphError
from app.core.ids import generate_request_id
from app.core.logging import logger, request_id_ctx


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Assigns and propagates a stable request ID across context and response headers."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Accept client-provided trace ID if valid, or generate a new one
        client_req_id = request.headers.get("X-Request-ID")
        req_id = client_req_id if client_req_id else generate_request_id()

        # Set context variable for structured logging
        token = request_id_ctx.set(req_id)
        request.state.request_id = req_id

        start_time = time.perf_counter()
        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            response.headers["X-Request-ID"] = req_id

            logger.info(
                f"{request.method} {request.url.path} returned "
                f"{response.status_code} in {duration_ms:.2f}ms"
            )
            return response
        except ClinGraphError as cge:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.warning(
                f"{request.method} {request.url.path} failed with {cge.code}: "
                f"{cge.message} ({duration_ms:.2f}ms)"
            )
            return JSONResponse(
                status_code=cge.status_code,
                content={
                    "error": {
                        "code": cge.code,
                        "message": cge.message,
                        "request_id": req_id,
                        "details": cge.details,
                    }
                },
                headers={"X-Request-ID": req_id},
            )
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                f"Unhandled exception during {request.method} {request.url.path} "
                f"({duration_ms:.2f}ms): {str(exc)}",
                exc_info=True,
            )
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": "INTERNAL_SERVER_ERROR",
                        "message": "An unexpected internal server error occurred.",
                        "request_id": req_id,
                        "details": {},
                    }
                },
                headers={"X-Request-ID": req_id},
            )
        finally:
            request_id_ctx.reset(token)
