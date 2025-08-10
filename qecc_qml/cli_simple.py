#!/usr/bin/env python3
"""
Simple CLI for QECC-QML Framework
"""

import sys
import json
import time
import http.server
import socketserver
from qecc_qml import __version__


def serve_command(host='127.0.0.1', port=8000):
    """Start simple HTTP server for health checks."""
    
    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/health':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {
                    "status": "healthy",
                    "version": __version__,
                    "timestamp": time.time()
                }
                self.wfile.write(json.dumps(response).encode())
            elif self.path == '/ready':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {
                    "status": "ready",
                    "version": __version__,
                    "timestamp": time.time()
                }
                self.wfile.write(json.dumps(response).encode())
            else:
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                html_content = f'''
                <!DOCTYPE html>
                <html>
                <head><title>QECC-QML Framework</title></head>
                <body>
                    <h1>🚀 QECC-QML Framework</h1>
                    <p>Version: {__version__}</p>
                    <p>Status: Running</p>
                    <ul>
                        <li><a href="/health">Health Check</a></li>
                        <li><a href="/ready">Readiness Check</a></li>
                    </ul>
                </body>
                </html>
                '''
                self.wfile.write(html_content.encode())
    
    print(f"🚀 Starting QECC-QML server on {host}:{port}")
    print(f"Version: {__version__}")
    
    with socketserver.TCPServer((host, port), Handler) as httpd:
        print(f"✅ Server running at http://{host}:{port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Server stopped")


def status_command():
    """Show framework status."""
    print("📊 QECC-QML Framework Status")
    print("=" * 30)
    
    print(f"Version: {__version__}")
    print(f"Python: {sys.version.split()[0]}")
    
    # Check dependencies
    dependencies = [
        ("NumPy", "numpy"),
        ("Qiskit", "qiskit"),
        ("Qiskit Aer", "qiskit_aer"),
    ]
    
    print("\n📦 Dependencies:")
    for name, module in dependencies:
        try:
            __import__(module)
            print(f"  ✅ {name}: Available")
        except ImportError:
            print(f"  ❌ {name}: Not installed")
    
    print("\n✅ Framework is operational")


def main():
    """Simple CLI main function."""
    if len(sys.argv) < 2:
        print("Usage: python cli_simple.py <command>")
        print("Commands: serve, status")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == 'serve':
        serve_command()
    elif command == 'status':
        status_command()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == '__main__':
    main()