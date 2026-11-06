"""HTTP transport helpers shared by authenticated model adapters."""

from urllib import request


class _RejectRedirects(request.HTTPRedirectHandler):
    """Never forward an adapter request (or its bearer token) to a new URL."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def open_without_redirect(req: request.Request, *, timeout: float):
    """Open one explicitly configured endpoint without implicit redirects."""

    return request.build_opener(_RejectRedirects()).open(req, timeout=timeout)
