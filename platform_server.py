"""Serve the AD-L local experiment platform on loopback."""

import argparse
import errno
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import secrets
import threading
from urllib.parse import urlsplit
import webbrowser
from urllib.request import build_opener, ProxyHandler

from ad_l.platform_service import PlatformService
from ad_l.roles import ROLE_CYCLE

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"


def make_handler(service, token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def _reply(self, status, value, *, content_type="application/json; charset=utf-8", filename=None):
            payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode() if content_type.startswith("application/json") else value
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(payload)

        def _local(self):
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if self.headers.get("Host") not in allowed:
                self._reply(403, {"error": "仅支持本机访问"})
                return False
            origin = self.headers.get("Origin")
            if origin and origin not in {f"http://{host}" for host in allowed}:
                self._reply(403, {"error": "请求来源无效"})
                return False
            return True

        def do_GET(self):
            if not self._local():
                return
            path = urlsplit(self.path).path
            try:
                if path == "/api/health":
                    self._reply(200, {"app": "ad-l-platform", "version": 1})
                elif path == "/api/bootstrap":
                    self._reply(200, {**service.store.public(), "token": token, "datasets": service.dataset_catalog(),
                                      "roles": list(ROLE_CYCLE), "jobs": service.history(), "checkpoints": service.checkpoints()})
                elif path == "/api/jobs":
                    self._reply(200, {"jobs": service.history(), "checkpoints": service.checkpoints()})
                elif re.fullmatch(r"/api/jobs/[a-f0-9]{32}", path):
                    self._reply(200, service.public_job(path.rsplit("/", 1)[1]))
                elif re.fullmatch(r"/api/jobs/[a-f0-9]{32}/export", path):
                    identity = path.split("/")[3]
                    self._reply(200, service.export(identity), filename=f"ad-l-{identity}.json")
                elif re.fullmatch(r"/api/jobs/[a-f0-9]{32}/checkpoint", path):
                    identity = path.split("/")[3]
                    service.public_job(identity)
                    checkpoint = service.state_dir / "jobs" / identity / "checkpoint.pt"
                    self._reply(200, checkpoint.read_bytes(), content_type="application/octet-stream", filename=f"topology-{identity}.pt")
                elif path in {"/", "/index.html", "/app.js", "/style.css"}:
                    filename = "index.html" if path == "/" else path[1:]
                    mime = {"index.html": "text/html; charset=utf-8", "app.js": "text/javascript; charset=utf-8", "style.css": "text/css; charset=utf-8"}[filename]
                    self._reply(200, (WEB / filename).read_bytes(), content_type=mime)
                else:
                    self._reply(404, {"error": "页面不存在"})
            except (ValueError, FileNotFoundError):
                self._reply(404, {"error": "请求的记录不存在"})
            except Exception as error:
                self._reply(500, {"error": f"读取失败（{type(error).__name__}）"})

        def do_POST(self):
            if not self._local():
                return
            if not hmac.compare_digest(self.headers.get("X-ADL-Token", ""), token):
                self._reply(403, {"error": "页面会话已过期，请刷新页面"})
                return
            if self.headers.get_content_type() != "application/json":
                self._reply(415, {"error": "请求格式须为 JSON"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1024 * 1024:
                    raise ValueError("请求大小无效")
                body = json.loads(self.rfile.read(size))
                path = urlsplit(self.path).path
                if path == "/api/config":
                    result = service.store.save(body["config"], body.get("keys"), body.get("clear_keys"))
                elif path == "/api/import-config":
                    result = service.import_local_config()
                elif path == "/api/preview":
                    result = service.preview(body["config"])
                elif path == "/api/plan":
                    result = service.plan(body["config"], body["kind"])
                elif path == "/api/checkpoint-settings":
                    result = service.checkpoint_settings(body["checkpoint"])
                elif path == "/api/jobs":
                    result = service.submit(body["kind"], body["config"], allow_real_api=body.get("allow_real_api") is True)
                elif re.fullmatch(r"/api/jobs/[a-f0-9]{32}/cancel", path):
                    result = service.cancel(path.split("/")[3])
                else:
                    self._reply(404, {"error": "接口不存在"})
                    return
                self._reply(200, result)
            except (ValueError, TypeError, KeyError, FileNotFoundError) as error:
                message = str(error) if isinstance(error, ValueError) else "配置字段或本地文件无效"
                self._reply(400, {"error": message})
            except Exception as error:
                self._reply(500, {"error": f"操作失败（{type(error).__name__}）"})
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), BaseHTTPRequestHandler)
    except OSError as error:
        if error.errno != errno.EADDRINUSE:
            raise
        url = f"http://127.0.0.1:{args.port}"
        with build_opener(ProxyHandler({})).open(f"{url}/api/health", timeout=3) as response:
            if json.load(response).get("app") != "ad-l-platform":
                raise SystemExit("本地端口被其他程序使用，请关闭占用 8765 端口的程序")
        if not args.no_browser:
            webbrowser.open(url)
        print(f"AD-L 评测平台已在运行 {url}", flush=True)
        return
    service = PlatformService(ROOT)
    server.RequestHandlerClass = make_handler(service, secrets.token_urlsafe(32))
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"AD-L 评测平台已启动 {url}", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        if service.active:
            service.cancel(service.active)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
