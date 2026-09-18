"""
Debug/visualization-only path for a BeamNG session.

Connects to a running BeamNG instance exactly like a training collector
(GameInstanceManager + VehicleInterface) and, instead of learning, streams the
extracted per-step state to a local web page:

    * the 160x120 grayscale camera frame as delivered to the conv head
    * the full float feature vector that feeds the network
    * a handful of live metadata numbers (zone, distance, sim time, action)

No tensorboard logging is created and nothing is saved to disk.

Usage:
    python scripts/debug_state_viz.py [--port 8478] [--http-port 8080]
"""

import argparse
import base64
import json
import queue
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

STATE_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Linesight &mdash; BeamNG state debug view</title>
<style>
  body { font-family: Consolas, "Courier New", monospace; background: #0f0f0f; color: #ddd; margin: 0; }
  header { padding: 8px 16px; background: #1b1b1b; border-bottom: 1px solid #333; position: sticky; top: 0; }
  h1 { font-size: 15px; display: inline; margin: 0 16px 0 0; color: #eee; }
  #status { margin-right: 12px; font-size: 12px; }
  #status.red { color: #f88; }
  #status.ok { color: #7d7; }
  #meta { color: #8cf; font-size: 12px; }
  #main { display: flex; gap: 16px; padding: 14px; align-items: flex-start; }
  #left { flex: 0 0 auto; }
  #frame { background: #000; border: 1px solid #333; image-rendering: pixelated; width: 480px; height: 360px; }
  #fps { color: #666; font-size: 11px; margin-top: 6px; text-align: center; }
  #panel { flex: 1 1 auto; min-width: 480px; }
  #scroll { max-height: 84vh; overflow-y: auto; }
  table { border-collapse: collapse; width: 100%; font-size: 11px; }
  th { text-align: left; color: #999; padding: 2px 6px; border-bottom: 1px solid #444; position: sticky; top: 0; background: #0f0f0f; }
  td { padding: 1px 6px; border-bottom: 1px solid #202020; white-space: nowrap; }
  td.idx { color: #555; width: 40px; text-align: right; }
  td.name { color: #9c9; }
  td.val { color: #fff; text-align: right; font-variant-numeric: tabular-nums; }
  details { margin-bottom: 8px; }
  summary { cursor: pointer; color: #fc8; font-size: 12px; padding: 2px 0; }
</style>
</head>
<body>
<header>
  <h1>Linesight &mdash; BeamNG state debug view</h1>
  <span id="status" class="red">connecting&hellip;</span>
  <span id="meta"></span>
</header>
<div id="main">
  <div id="left">
    <canvas id="frame" width="160" height="120"></canvas>
    <div id="fps"></div>
  </div>
  <div id="panel">
    <div id="scroll">
      <table>
        <thead><tr><th>#</th><th>feature</th><th>value</th></tr></thead>
        <tbody id="tbody-main"></tbody>
      </table>
      <details open>
        <summary>Zone centers &mdash; next 40 VCPs in car frame (indices 62..181)</summary>
        <table>
          <thead><tr><th>#</th><th>feature</th><th>value</th></tr></thead>
          <tbody id="tbody-zones"></tbody>
        </table>
      </details>
    </div>
  </div>
</div>
<script>
const LABELS = __LABELS__;
const ZONE_START = __ZONE_START__;
const canvas = document.getElementById("frame");
const ctx = canvas.getContext("2d");
const imgData = ctx.createImageData(160, 120);

function addRow(tbody, i) {
  const tr = document.createElement("tr");
  const a = document.createElement("td"); a.className = "idx"; a.textContent = i;
  const b = document.createElement("td"); b.className = "name"; b.textContent = LABELS[i];
  if (LABELS[i].startsWith("zone_center")) tr.style.color = "#7ad";
  const c = document.createElement("td"); c.className = "val";
  tr.append(a, b, c);
  tbody.appendChild(tr);
  return c;
}

const cellsMain = [];
const tbodyMain = document.getElementById("tbody-main");
for (let i = 0; i < ZONE_START; i++) cellsMain.push(addRow(tbodyMain, i));
const cellsZones = [];
const tbodyZones = document.getElementById("tbody-zones");
for (let i = ZONE_START; i < LABELS.length; i++) cellsZones.push(addRow(tbodyZones, i));

const status = document.getElementById("status");
const metaEl = document.getElementById("meta");
const fpsEl = document.getElementById("fps");
const lastTimes = [];

const es = new EventSource("/stream");
es.onopen = () => { status.textContent = "live"; status.className = "ok"; };
es.onerror = () => { status.textContent = "reconnecting\u2026"; status.className = "red"; };
es.onmessage = (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.w === 160 && msg.h === 120) {
    const raw = atob(msg.frame);
    const buf = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) buf[i] = raw.charCodeAt(i);
    for (let i = 0, j = 0; i < buf.length; i++, j += 4) {
      imgData.data[j] = buf[i];
      imgData.data[j + 1] = buf[i];
      imgData.data[j + 2] = buf[i];
      imgData.data[j + 3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);
  }
  const fl = msg.floats;
  for (let i = 0; i < cellsMain.length; i++) cellsMain[i].textContent = fl[i].toFixed(4);
  for (let i = 0; i < cellsZones.length; i++) cellsZones[i].textContent = fl[ZONE_START + i].toFixed(5);
  const m = msg.meta;
  metaEl.textContent = "sim " + m.sim_s + "s | race " + m.race_ms + "ms | zone " + m.zone +
    " | dist " + m.dist_m + "m | speed " + m.speed_mps + "m/s | pos " + m.pos.join(", ") +
    " | action " + m.action + (m.greedy ? "" : " \u00b7explo");
  const now = performance.now();
  lastTimes.push(now);
  if (lastTimes.length > 120) lastTimes.shift();
  if (lastTimes.length > 1 && now - lastTimes[0] > 1000) {
    fpsEl.textContent = ((lastTimes.length - 1) / (now - lastTimes[0]) * 1000).toFixed(1) + " ev/s";
  }
};
</script>
</body>
</html>
"""


class StateBroadcaster:
    """Fan-out the latest extracted state to all connected SSE clients."""

    def __init__(self):
        self._subscribers = []
        self._lock = threading.Lock()
        self._latest = None

    def subscribe(self):
        subscriber = queue.Queue(maxsize=4)
        with self._lock:
            if self._latest is not None:
                subscriber.put(self._latest)
            self._subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber):
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

    def publish(self, payload: str):
        with self._lock:
            subscribers = list(self._subscribers)
            self._latest = payload
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(payload)
            except queue.Full:
                try:
                    subscriber.get_nowait()
                    subscriber.put_nowait(payload)
                except queue.Empty:
                    pass


class StatePageHandler(BaseHTTPRequestHandler):
    """Serve the static page and an SSE stream of extracted states."""

    broadcaster = None
    page_html = ""

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/":
            body = self.page_html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/stream":
            self._stream_states()
        else:
            self.send_response(404)
            self.end_headers()

    def _stream_states(self):
        subscriber = self.broadcaster.subscribe()
        self.protocol_version = "HTTP/1.1"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            while True:
                try:
                    payload = subscriber.get(timeout=15)
                    self.wfile.write(b"data: " + payload.encode("utf-8") + b"\n\n")
                    self.wfile.flush()
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        finally:
            self.broadcaster.unsubscribe(subscriber)


def copy_configuration_file():
    base_dir = Path(__file__).resolve().parents[1]
    shutil.copyfile(
        base_dir / "config_files" / "config.py",
        base_dir / "config_files" / "config_copy.py",
    )


def build_float_labels(config):
    """Name every element of the float feature vector, mirroring _build_float_inputs."""
    labels = ["temporal_mini_race (placeholder)"]
    for k in range(config.n_prev_actions_in_inputs):
        for name in ("increase_throttle", "increase_brake", "left", "right"):
            labels.append(f"prev_action[{k}].{name}")
    labels += ["throttle", "brake", "steering", "damage_overall"]
    labels += ["slip_fl", "slip_fr", "slip_rl", "slip_rr"]
    labels += ["downforce_fl", "downforce_fr", "downforce_rl", "downforce_rr"]
    labels += ["gearbox_state", "gear", "rpm", "gearbox_counter"]
    for wheel in ("fl", "fr", "rl", "rr"):
        for c in range(config.n_contact_material_physics_behavior_types):
            labels.append(f"material[{wheel}].class{c}")
    labels += ["angvel_car_x", "angvel_car_y", "angvel_car_z"]
    labels += ["vel_car_x", "vel_car_y", "vel_car_z"]
    labels += ["worldup_car_x", "worldup_car_y", "worldup_car_z"]

    zone_start_idx = len(labels)
    zone_offsets_m = (
        (np.arange(config.n_zone_centers_in_inputs) + 1) * config.one_every_n_zone_centers_in_inputs * config.distance_between_checkpoints
    )
    for offset_m in zone_offsets_m:
        for axis in ("x", "y", "z"):
            labels.append(f"zone_center[{offset_m:.1f}m].{axis}")

    labels += ["finish_distance_m", "is_freewheeling"]
    assert len(labels) == config.float_input_dim, f"label count {len(labels)} != float_input_dim {config.float_input_dim}"
    return labels, zone_start_idx


def build_payload(config, step):
    frame = np.asarray(step["frame"], dtype=np.uint8).reshape(-1)
    state = step["state"]
    velocity = np.asarray(state["velocity"], dtype=np.float32)
    payload = {
        "frame": base64.b64encode(frame.tobytes()).decode("ascii"),
        "w": int(config.W_downsized),
        "h": int(config.H_downsized),
        "meta": {
            "sim_s": round(float(state["sim_time"]), 3),
            "race_ms": int(step["race_time_ms"]),
            "zone": int(step["current_zone_idx"]),
            "dist_m": round(float(step["distance_since_track_begin"]), 2),
            "speed_mps": round(float(np.linalg.norm(velocity)), 2),
            "pos": [round(float(v), 2) for v in state["position"]],
            "action": int(step["action_idx"]),
            "greedy": bool(step["action_was_greedy"]),
        },
        "floats": [round(float(v), 5) for v in step["floats"]],
    }
    return json.dumps(payload, separators=(",", ":"))


def make_fixed_forward_policy(config):
    def policy(frame, floats):
        q_values = np.zeros(len(config.inputs), dtype=np.float32)
        q_values[config.action_forward_idx] = 1.0
        return config.action_forward_idx, True, 1.0, q_values

    return policy


def run_state_stream(broadcaster, config, port):
    from trackmania_rl.tmi_interaction.game_instance_manager import GameInstanceManager

    manager = GameInstanceManager(
        game_spawning_lock=threading.Lock(),
        running_speed=config.running_speed,
        run_steps_per_action=config.tm_engine_step_per_action,
        max_overall_duration_ms=config.cutoff_rollout_if_race_not_finished_within_duration_ms,
        max_minirace_duration_ms=config.cutoff_rollout_if_no_vcp_passed_within_duration_ms,
        tmi_port=port,
    )
    policy = make_fixed_forward_policy(config)

    def on_state(**step):
        broadcaster.publish(build_payload(config, step))

    print(f"[debug_viz] vehicle interface on port {port}; streaming state. Ctrl+C to stop.")
    while True:
        try:
            manager.rollout(
                exploration_policy=policy,
                map_path="",
                zone_centers=None,
                update_network=None,
                on_state=on_state,
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 — top-level driver keeps running across game/interface crashes
            print(f"[debug_viz] rollout crashed ({exc!r}); reconnecting in 2s")
            manager.iface = None
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, help="Vehicle interface TCP port (default: base_tmi_port from config).")
    parser.add_argument("--http-port", type=int, default=8080, help="Local web UI port.")
    args = parser.parse_args()

    copy_configuration_file()
    from config_files import config_copy

    vehicle_port = args.port if args.port is not None else config_copy.base_tmi_port

    labels, zone_start = build_float_labels(config_copy)
    page_html = STATE_PAGE.replace("__LABELS__", json.dumps(labels)).replace("__ZONE_START__", str(zone_start))

    broadcaster = StateBroadcaster()
    StatePageHandler.broadcaster = broadcaster
    StatePageHandler.page_html = page_html

    server = ThreadingHTTPServer(("127.0.0.1", args.http_port), StatePageHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"[debug_viz] open http://127.0.0.1:{args.http_port}/ in your browser")

    run_state_stream(broadcaster, config_copy, vehicle_port)


if __name__ == "__main__":
    main()
