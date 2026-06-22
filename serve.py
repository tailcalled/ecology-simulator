#!/usr/bin/env python3
"""Local dev server that sets the COOP/COEP headers required for SharedArrayBuffer
(and therefore wasm threads). http://localhost is a secure context, so no HTTPS is needed.

Usage:  python3 serve.py [port]   (default 8080)
Then open http://localhost:8080/
"""
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".wasm": "application/wasm",
    }

    def end_headers(self):
        # Required for crossOriginIsolated === true (enables SharedArrayBuffer).
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        # Don't cache during development.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    httpd = HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Serving http://localhost:{PORT}/  (COOP/COEP enabled)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
