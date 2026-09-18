import socket
import struct
import time
import numpy as np
from enum import IntEnum
from beamngpy import BeamNGpy, Vehicle
from beamngpy.sensors import Camera

HOST = "127.0.0.1"

STATE_FIELDS = [
    ("sim_time", "f"),          # approximately 0.05, 0.10, 0.15, ... (in seconds)
    ("throttle", "f"),          # [0, 1]
    ("brake", "f"),             # [0, 1]
    ("steering", "f"),          # [-1, 1]
    ("gear", "i"),              # gear
    ("rpm", "i"),               # engine rpm
    ("x", "f"),                 # car position in the world (meters)
    ("y", "f"),                 #
    ("z", "f"),                 #
    ("vx", "f"),                # car velocity in the world (meters/sec)
    ("vy", "f"),                #
    ("vz", "f"),                #
    ("roll_angular", "f"),
    ("pitch_angular", "f"),
    ("yaw_angular", "f"),
    ("forward_vector_x", "f"),  # unit vector attached to car and pointing forward
    ("forward_vector_y", "f"),  #
    ("forward_vector_z", "f"),  #
    ("up_vector_x", "f"),       # unit vector attached to car and pointing up (z-up)
    ("up_vector_y", "f"),       #
    ("up_vector_z", "f"),       #
    ("right_vector_x", "f"),    # unit vector attached to car and pointing right
    ("right_vector_y", "f"),    #
    ("right_vector_z", "f"),    #
    ("fl_slip", "f"),           # per-wheel skidding amounts
    ("fr_slip", "f"),           #
    ("rl_slip", "f"),           #
    ("rr_slip", "f"),           #
    ("fl_downforce", "f"),      # per-wheel down forces
    ("fr_downforce", "f"),      #
    ("rl_downforce", "f"),      #
    ("rr_downforce", "f"),      #
    ("fl_material", "i"),       # per-wheel contact materials
    ("fr_material", "i"),       #
    ("rl_material", "i"),       #
    ("rr_material", "i"),       #
    ("damage_overall", "f"),
]
STATE_STRUCT = struct.Struct("<" + "".join(fmt for _, fmt in STATE_FIELDS))

class Message(IntEnum):
    RESET = 1
    RESET_TELEPORT = 2
    SET_ACTION = 3
    STEP_COMPLETE = 100

class VehicleInterface:
    def __init__(self, PORT):
        game = BeamNGpy(
            "localhost",
            25252,
            home='C:/Users/ouasd/Desktop/BeamNG.tech.v0.38.5.0',
            user='C:/Users/ouasd/AppData/Local/BeamNG/BeamNG.tech',
        )
        game.open(launch=False) # connect to existing instance
        veh_idx = PORT-8478
        veh = game.get_current_vehicles()[f"some_vehicle_{veh_idx}"]

        self.camera = Camera(
            name=f"camera{veh_idx}", bng=game, vehicle=veh,
            requested_update_time=0.05, # requested_update_time=-1,
            resolution=(160, 120), near_far_planes=(0.05, 250.0),
            pos=(0, -1.25, 1.25), dir=(0, 0, 0),
            is_streaming=True, is_using_shared_memory=True,
            is_render_annotations=False, is_render_depth=True,
        )

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(
            socket.IPPROTO_TCP,
            socket.TCP_NODELAY,
            1
        )
        self.server.bind((HOST, PORT))
        self.server.listen(1)
        print("Waiting for Lua...")
        self.sock, addr = self.server.accept()
        print("Connected:", addr)

    def get_visual_input(self, vis_type="color"):
        raw_readings = self.camera.stream_raw()
        if vis_type == "color":
            vis_color = np.frombuffer(raw_readings['colour'], dtype=np.uint8).reshape((120, 160, 4))
            vis_color[:, :, 3] = 255
            vis_color = np.round(vis_color[:, :, :3].mean(axis=-1)).astype(np.uint8)
            vis_gray = np.expand_dims(vis_color, axis=0)
            return vis_gray
        elif vis_type == "depth":
            vis_depth = np.frombuffer(raw_readings['depth'], dtype=np.float32).reshape((120, 160))
            vis_depth = np.expand_dims(vis_depth, axis=0)
            return vis_depth

    def get_state(self):
        values = STATE_STRUCT.unpack(self._recv_exact(STATE_STRUCT.size))
        state = dict(zip((name for name, _ in STATE_FIELDS), values))
        
        state["position"]           = (state.pop("x"),                  state.pop("y"),                 state.pop("z"))
        state["velocity"]           = (state.pop("vx"),                 state.pop("vy"),                state.pop("vz"))
        state["angular_velocity"]   = (state.pop("roll_angular"),       state.pop("pitch_angular"),     state.pop("yaw_angular"))
        state["forward_vector"]     = (state.pop("forward_vector_x"),   state.pop("forward_vector_y"),  state.pop("forward_vector_z"))
        state["up_vector"]          = (state.pop("up_vector_x"),        state.pop("up_vector_y"),       state.pop("up_vector_z"))
        state["right_vector"]       = (state.pop("right_vector_x"),     state.pop("right_vector_y"),    state.pop("right_vector_z"))
        state["slips"]              = (state.pop("fl_slip"),        state.pop("fr_slip"),       state.pop("rl_slip"),       state.pop("rr_slip"))
        state["downforces"]         = (state.pop("fl_downforce"),   state.pop("fr_downforce"),  state.pop("rl_downforce"),  state.pop("rr_downforce"))
        state["materials"]          = (state.pop("fl_material"),    state.pop("fr_material"),   state.pop("rl_material"),   state.pop("rr_material"))
        return state

    def reset(self):
        self.sock.sendall(
            struct.pack("<I", Message.RESET)
        )

    def reset_and_teleport(self, pos, rot):
        self.sock.sendall(
            struct.pack("<I6f", Message.RESET_TELEPORT, *pos, *rot)
        )

    def send_action(self, left, right, center, throttle, brake, reset_throttle, reset_brake):
        self.sock.sendall(
            struct.pack("<I7B", Message.SET_ACTION, left, right, center, throttle, brake, reset_throttle, reset_brake)
        )

    def step_complete(self):
        self.sock.sendall(
            struct.pack("<I", Message.STEP_COMPLETE)
        )

    def _recv_exact(self, n):
        data = bytearray()
        while len(data) < n:
            chunk = self.sock.recv(n - len(data))
            if not chunk:
                raise ConnectionError
            data.extend(chunk)
        return data