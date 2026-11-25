#!/usr/bin/env python3
"""Web 服务（纯标准库）：县域比率分层估计与总量校准。

    python3 web/server.py [--port 8000]

接口（全部为幂等 GET，便于核查与复现）：
    /                              前端页面
    /api/state?scenario=X&locked=C01,C03   估计+校准+验证汇总
    /api/validation                整地区留出验证明细
    /api/export/estimates.csv?...  明细导出（与 CLI 同一代码路径）
    /api/export/summary.json?...
"""

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from countyrate.data import generate_dataset, default_locked_ids
from countyrate.pipeline import compute_estimates, compute_validation

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

DATASET = generate_dataset()
VALIDATION = compute_validation(DATASET)  # 与场景无关，启动时算一次
DEFAULT_LOCKED = default_locked_ids(DATASET)

CONTENT_TYPES = {".html": "text/html; charset=utf-8",
                 ".js": "text/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8"}


def _parse_locked(qs):
    raw = qs.get("locked", [None])[0]
    if raw is None:
        return list(DEFAULT_LOCKED)
    if raw == "":
        return []
    return [s for s in raw.split(",") if s]


def _estimates_csv(result):
    import io
    buf = io.StringIO()
    cols = ["id", "name", "region", "population", "n_sample", "n_eff", "deff",
            "raw_rate", "confirmed_count", "shrunk_rate", "shrunk_lo", "shrunk_hi",
            "shrink_weight_B", "extrapolated", "cal_rate", "cal_lo", "cal_hi",
            "locked", "clamped", "true_rate"]
    buf.write(",".join(cols) + "\n")
    PCT = lambda x: "" if x is None else f"{x:.9f}"
    for c in result["counties"]:
        buf.write(",".join([
            c["id"], c["name"], c["region"], str(c["population"]),
            str(c["n_sample"]), str(c["n_eff"]), str(c["deff"] or ""),
            PCT(c["raw_rate"]), str(c["confirmed_count"] or ""),
            PCT(c["shrunk_rate"]), PCT(c["shrunk_lo"]), PCT(c["shrunk_hi"]),
            f"{c['shrink_weight_B']:.4f}", str(int(c["extrapolated"])),
            PCT(c["cal_rate"]), PCT(c["cal_lo"]), PCT(c["cal_hi"]),
            str(int(c["locked"])), c["clamped"] or "", PCT(c["true_rate"]),
        ]) + "\n")
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    server_version = "CountyRate/0.1"

    def _send(self, body, content_type="application/json; charset=utf-8", status=200):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, status=200):
        self._send(json.dumps(obj, ensure_ascii=False), status=status)

    def do_GET(self):
        url = urlparse(self.path)
        # keep_blank_values: "locked="（空串）表示"全部解锁"，区别于缺省（默认锁定）
        path, qs = url.path, parse_qs(url.query, keep_blank_values=True)
        try:
            if path == "/":
                self._serve_static("index.html")
            elif path.startswith("/static/"):
                self._serve_static(path[len("/static/"):])
            elif path == "/api/state":
                self._api_state(qs)
            elif path == "/api/validation":
                self._json(VALIDATION)
            elif path == "/api/export/estimates.csv":
                result = self._compute(qs)
                self._send(_estimates_csv(result),
                           "text/csv; charset=utf-8-sig")
            elif path == "/api/export/summary.json":
                result = self._compute(qs)
                self._json({"estimates": result, "validation": VALIDATION})
            else:
                self._json({"error": "not found"}, status=404)
        except ValueError as e:
            self._json({"error": str(e)}, status=400)
        except Exception as e:  # 服务稳健性：错误以 JSON 返回而非崩溃
            self._json({"error": f"{type(e).__name__}: {e}"}, status=500)

    def _compute(self, qs):
        scenario = qs.get("scenario", ["baseline"])[0]
        if scenario not in DATASET["scenarios"]:
            raise ValueError(f"未知场景: {scenario}")
        return compute_estimates(DATASET, scenario, _parse_locked(qs))

    def _api_state(self, qs):
        result = self._compute(qs)
        self._json({
            "estimates": result,
            "validation_overall": VALIDATION["overall"],
            "scenarios": {k: v for k, v in DATASET["scenarios"].items()},
            "default_locked": DEFAULT_LOCKED,
        })

    def _serve_static(self, name):
        safe = os.path.normpath(name).lstrip(os.sep)
        full = os.path.join(STATIC_DIR, safe)
        if not full.startswith(STATIC_DIR) or not os.path.isfile(full):
            self._json({"error": "not found"}, status=404)
            return
        ext = os.path.splitext(full)[1]
        with open(full, "rb") as f:
            self._send(f.read(), CONTENT_TYPES.get(ext, "application/octet-stream"))

    def log_message(self, fmt, *args):
        sys.stderr.write("[web] %s\n" % (fmt % args))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"县域比率分析应用: http://127.0.0.1:{args.port}/")
    print(f"默认锁定 {len(DEFAULT_LOCKED)} 个已确认县; "
          f"场景: {', '.join(DATASET['scenarios'])}")
    server.serve_forever()


if __name__ == "__main__":
    main()
