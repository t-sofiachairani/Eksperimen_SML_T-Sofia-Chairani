import time
import requests
from prometheus_client import start_http_server, Counter, Histogram

MODEL_PING = "http://kue_model:8080/ping"

REQ_COUNT = Counter("inference_requests_total", "Total requests", ["status"])
REQ_LAT = Histogram("inference_latency_seconds", "Latency (seconds)")

def loop():
    while True:
        t0 = time.time()
        try:
            r = requests.get(MODEL_PING, timeout=10)
            dt = time.time() - t0
            REQ_LAT.observe(dt)

            if 200 <= r.status_code < 300:
                REQ_COUNT.labels(status="success").inc()
            else:
                REQ_COUNT.labels(status="fail").inc()
        except Exception:
            dt = time.time() - t0
            REQ_LAT.observe(dt)
            REQ_COUNT.labels(status="error").inc()

        time.sleep(2)

if __name__ == "__main__":
    start_http_server(8000)  # exporter metrics di :8000/metrics
    loop()
