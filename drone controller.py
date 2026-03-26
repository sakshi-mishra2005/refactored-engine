import cv2
import numpy as np
import zmq
import time
import argparse
import threading
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil

# ── Optional picamera2 or fallback to USB camera ────────────────────────
try:
    from picamera2 import Picamera2
    USE_PICAMERA = True
except ImportError:
    USE_PICAMERA = False

# ════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  —  change these to match your setup
# ════════════════════════════════════════════════════════════════════════
PIXHAWK_PORT     = "/dev/ttyUSB0"   # or "/dev/ttyAMA0" for UART
PIXHAWK_BAUD     = 57600
LIDAR_PORT       = "/dev/ttyUSB1"   # change if LiDAR is on different USB
FRAME_W          = 320
FRAME_H          = 240
ZMQ_PORT         = 5555             # port to stream video to laptop
OBSTACLE_DIST_CM = 80               # stop if anything closer than 80cm
MATCH_THRESHOLD  = 12               # min ORB matches to confirm target found
TARGET_HOVER_ALT = 1.5              # metres to hover over target once found
BATTERY_LAND_PCT = 20               # auto-land if battery below 20%

# ════════════════════════════════════════════════════════════════════════
#  CAMERA SETUP
# ════════════════════════════════════════════════════════════════════════
def open_camera():
    if USE_PICAMERA:
        cam = Picamera2()
        cfg = cam.create_preview_configuration(
            main={"size": (FRAME_W, FRAME_H), "format": "RGB888"}
        )
        cam.configure(cfg)
        cam.start()
        time.sleep(0.5)
        return cam
    else:
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        return cam

def read_frame(cam):
    if USE_PICAMERA:
        rgb = cam.capture_array()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, frame = cam.read()
    return frame if ok else None

# ════════════════════════════════════════════════════════════════════════
#  IMAGE MATCHING  —  ORB feature detector
# ════════════════════════════════════════════════════════════════════════
class TargetMatcher:
    """
    Loads a target image and matches it against live camera frames.
    Uses ORB (fast, works on Pi) + BFMatcher.
    Returns (matched, centre_x, centre_y) where centre is in frame pixels.
    """
    def __init__(self, target_path: str):
        self.orb     = cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.target_kp   = None
        self.target_desc = None
        self.load_target(target_path)

    def load_target(self, path: str):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] Target image not found: {path}")
            return
        self.target_kp, self.target_desc = self.orb.detectAndCompute(img, None)
        print(f"[Match] Target loaded — {len(self.target_kp)} keypoints")

    def update_target_from_bytes(self, raw_bytes):
        """Called when laptop sends a new target image over ZMQ."""
        buf  = np.frombuffer(raw_bytes, dtype=np.uint8)
        img  = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return
        self.target_kp, self.target_desc = self.orb.detectAndCompute(img, None)
        print(f"[Match] New target received — {len(self.target_kp)} keypoints")

    def match(self, frame_bgr):
        """
        Returns (found: bool, cx: int, cy: int, annotated_frame)
        cx, cy = pixel centre of matched region in the frame.
        """
        if self.target_desc is None:
            return False, 0, 0, frame_bgr

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kp, desc = self.orb.detectAndCompute(gray, None)

        if desc is None or len(desc) < 5:
            return False, 0, 0, frame_bgr

        matches = self.matcher.match(self.target_desc, desc)
        matches = sorted(matches, key=lambda m: m.distance)
        good    = [m for m in matches if m.distance < 60]

        found = len(good) >= MATCH_THRESHOLD
        cx, cy = 0, 0

        if found:
            # find centre of matched keypoints in the frame
            pts = np.array([kp[m.trainIdx].pt for m in good])
            cx  = int(pts[:, 0].mean())
            cy  = int(pts[:, 1].mean())
            cv2.circle(frame_bgr, (cx, cy), 20, (0, 255, 0), 3)
            cv2.putText(frame_bgr, f"TARGET FOUND ({len(good)} pts)",
                (cx - 80, cy - 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)

        return found, cx, cy, frame_bgr

# ════════════════════════════════════════════════════════════════════════
#  PID CONTROLLER  —  converts pixel error → velocity command
# ════════════════════════════════════════════════════════════════════════
class PIDController:
    """
    Simple PID — takes error in pixels, outputs velocity in m/s.
    One PID for X (left/right), one for Y (forward/back).
    """
    def __init__(self, kp=0.003, ki=0.0001, kd=0.001, max_out=0.5):
        self.kp      = kp
        self.ki      = ki
        self.kd      = kd
        self.max_out = max_out
        self._integral  = 0.0
        self._prev_error = 0.0
        self._last_time  = time.time()

    def compute(self, error: float) -> float:
        now = time.time()
        dt  = now - self._last_time
        if dt <= 0:
            dt = 0.01

        self._integral   += error * dt
        derivative        = (error - self._prev_error) / dt
        output            = (self.kp * error +
                             self.ki * self._integral +
                             self.kd * derivative)

        self._prev_error = error
        self._last_time  = now
        # clamp output to max velocity
        return max(-self.max_out, min(self.max_out, output))

    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0

# ════════════════════════════════════════════════════════════════════════
#  LIDAR READER  —  runs in background thread
# ════════════════════════════════════════════════════════════════════════
class LidarReader:
    """
    Reads distance from RPLidar in a background thread.
    self.min_distance_cm = closest object seen in last scan.
    """
    def __init__(self, port: str):
        self.min_distance_cm = 9999
        self._running = False
        self._port    = port
        self._thread  = None

    def start(self):
        try:
            from rplidar import RPLidar
            self._lidar   = RPLidar(self._port)
            self._running = True
            self._thread  = threading.Thread(
                target=self._read_loop, daemon=True)
            self._thread.start()
            print(f"[LiDAR] Started on {self._port}")
        except Exception as e:
            print(f"[LiDAR] Could not start: {e} — obstacle avoidance disabled")

    def _read_loop(self):
        try:
            for scan in self._lidar.iter_scans():
                if not self._running:
                    break
                distances = [m[2] for m in scan if m[2] > 0]
                if distances:
                    self.min_distance_cm = min(distances) / 10.0
        except Exception as e:
            print(f"[LiDAR] Read error: {e}")

    def stop(self):
        self._running = False
        if hasattr(self, "_lidar"):
            self._lidar.stop()
            self._lidar.disconnect()

# ════════════════════════════════════════════════════════════════════════
#  MAVLINK HELPERS  —  send velocity commands to Pixhawk
# ════════════════════════════════════════════════════════════════════════
def send_velocity(vehicle, vx, vy, vz=0):
    """
    Send velocity command in body frame (drone's own forward/right/up axes).
    vx = forward/back  (positive = forward)
    vy = left/right    (positive = right)
    vz = up/down       (positive = DOWN in NED frame)
    """
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,                                      # time_boot_ms
        0, 0,                                   # target system, component
        mavutil.mavlink.MAV_FRAME_BODY_NED,     # body frame
        0b0000111111000111,                     # ignore position, keep velocity
        0, 0, 0,                                # x, y, z position (ignored)
        vx, vy, vz,                             # vx, vy, vz velocity m/s
        0, 0, 0,                                # acceleration (ignored)
        0, 0                                    # yaw, yaw_rate (ignored)
    )
    vehicle.send_mavlink(msg)

def arm_and_takeoff(vehicle, target_altitude_m: float):
    """Arms the drone and climbs to target_altitude_m metres."""
    print("[Flight] Running pre-arm checks...")
    while not vehicle.is_armable:
        print("  Waiting for drone to be armable...")
        time.sleep(1)

    print("[Flight] Arming motors...")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:
        print("  Waiting for arming...")
        time.sleep(1)

    print(f"[Flight] Taking off to {target_altitude_m}m...")
    vehicle.simple_takeoff(target_altitude_m)

    # wait until altitude is reached (within 95%)
    while True:
        alt = vehicle.location.global_relative_frame.alt
        print(f"  Altitude: {alt:.1f}m")
        if alt >= target_altitude_m * 0.95:
            print("[Flight] Target altitude reached!")
            break
        time.sleep(0.5)

def safe_land(vehicle):
    print("[Flight] Landing...")
    vehicle.mode = VehicleMode("LAND")
    while vehicle.armed:
        time.sleep(1)
    print("[Flight] Landed and disarmed.")

# ════════════════════════════════════════════════════════════════════════
#  ZMQ SETUP  —  video out to laptop, target image in from laptop
# ════════════════════════════════════════════════════════════════════════
def setup_zmq(port: int):
    ctx = zmq.Context()

    # publisher — sends video frames to laptop
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://*:{port}")

    # subscriber — receives target image from laptop
    sub = ctx.socket(zmq.SUB)
    sub.bind(f"tcp://*:{port + 1}")
    sub.setsockopt_string(zmq.SUBSCRIBE, "TARGET:")
    sub.setsockopt(zmq.RCVTIMEO, 10)   # non-blocking, 10ms timeout

    print(f"[ZMQ] Video stream on port {port}")
    print(f"[ZMQ] Listening for target image on port {port + 1}")
    return ctx, pub, sub

# ════════════════════════════════════════════════════════════════════════
#  MAIN FLIGHT LOOP
# ════════════════════════════════════════════════════════════════════════
def main(target_path: str, altitude: float, pixhawk_port: str):

    # ── connect to Pixhawk ───────────────────────────────────────────────
    print(f"[MAVLink] Connecting to Pixhawk on {pixhawk_port}...")
    vehicle = connect(pixhawk_port, baud=PIXHAWK_BAUD, wait_ready=True)
    print(f"[MAVLink] Connected!  Battery: {vehicle.battery.level}%")

    # ── start LiDAR ──────────────────────────────────────────────────────
    lidar = LidarReader(LIDAR_PORT)
    lidar.start()

    # ── open camera ──────────────────────────────────────────────────────
    cam = open_camera()

    # ── set up image matcher and PID ─────────────────────────────────────
    matcher = TargetMatcher(target_path)
    pid_x   = PIDController(kp=0.003, max_out=0.4)  # left/right
    pid_y   = PIDController(kp=0.003, max_out=0.4)  # forward/back

    # ── set up ZMQ ───────────────────────────────────────────────────────
    ctx, pub, sub = setup_zmq(ZMQ_PORT)

    # ── take off ─────────────────────────────────────────────────────────
    arm_and_takeoff(vehicle, altitude)

    print("\n[Flight] Searching for target — send image from laptop to update target")
    print("[Flight] Press Ctrl+C to land immediately\n")

    target_found_time = None
    frame_count       = 0

    try:
        while True:

            # ── safety: battery check ────────────────────────────────────
            batt = vehicle.battery.level
            if batt is not None and batt < BATTERY_LAND_PCT:
                print(f"[SAFETY] Battery low ({batt}%) — landing!")
                break

            # ── safety: obstacle check ───────────────────────────────────
            if lidar.min_distance_cm < OBSTACLE_DIST_CM:
                print(f"[SAFETY] Obstacle at {lidar.min_distance_cm:.0f}cm — stopping!")
                send_velocity(vehicle, 0, 0, 0)
                time.sleep(0.1)
                continue

            # ── read camera frame ─────────────────────────────────────────
            frame = read_frame(cam)
            if frame is None:
                continue
            frame_count += 1

            # ── check for new target from laptop ─────────────────────────
            try:
                raw = sub.recv()
                # message format: b"TARGET:<jpeg bytes>"
                img_bytes = raw[len(b"TARGET:"):]
                matcher.update_target_from_bytes(img_bytes)
                pid_x.reset()
                pid_y.reset()
                target_found_time = None
            except zmq.Again:
                pass  # no new target — that's fine

            # ── image matching ────────────────────────────────────────────
            found, cx, cy, annotated = matcher.match(frame)

            if found:
                # error = how far target centre is from frame centre
                err_x = cx - (FRAME_W // 2)   # positive = target is to the right
                err_y = cy - (FRAME_H // 2)   # positive = target is below centre

                # PID converts pixel error → velocity command
                vel_y =  pid_x.compute(err_x)  # right/left
                vel_x = -pid_y.compute(err_y)  # forward/back (camera y = drone x)

                send_velocity(vehicle, vel_x, vel_y, 0)

                # add HUD to frame
                cv2.putText(annotated,
                    f"err=({err_x:+d},{err_y:+d})  vel=({vel_x:+.2f},{vel_y:+.2f})",
                    (6, FRAME_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                # if centred within 20px for 2 seconds → we are above target!
                if abs(err_x) < 20 and abs(err_y) < 20:
                    if target_found_time is None:
                        target_found_time = time.time()
                    elif time.time() - target_found_time > 2.0:
                        print("[Flight] Hovering over target! Mission complete.")
                        send_velocity(vehicle, 0, 0, 0)
                        time.sleep(5)
                        break
                else:
                    target_found_time = None

            else:
                # no target — hover in place (send zero velocity)
                send_velocity(vehicle, 0, 0, 0)
                cv2.putText(annotated, "Searching...",
                    (6, FRAME_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

            # ── add telemetry overlay ─────────────────────────────────────
            alt_now = vehicle.location.global_relative_frame.alt
            cv2.putText(annotated,
                f"Alt:{alt_now:.1f}m  Batt:{batt}%  Obs:{lidar.min_distance_cm:.0f}cm",
                (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            # ── stream frame to laptop ────────────────────────────────────
            _, buf = cv2.imencode(".jpg", annotated,
                                  [cv2.IMWRITE_JPEG_QUALITY, 60])
            pub.send(buf.tobytes())

    except KeyboardInterrupt:
        print("\n[Flight] Ctrl+C — landing now")

    finally:
        safe_land(vehicle)
        lidar.stop()
        if USE_PICAMERA:
            cam.stop()
        else:
            cam.release()
        pub.close()
        sub.close()
        ctx.term()
        vehicle.close()
        print("[Flight] All systems shut down.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drone autonomous controller")
    parser.add_argument("--target",    type=str,   default="target.jpg",
                        help="Path to target image file (default: target.jpg)")
    parser.add_argument("--altitude",  type=float, default=2.0,
                        help="Takeoff altitude in metres (default: 2.0)")
    parser.add_argument("--pixhawk",   type=str,   default=PIXHAWK_PORT,
                        help=f"Pixhawk serial port (default: {PIXHAWK_PORT})")
    args = parser.parse_args()
    main(args.target, args.altitude, args.pixhawk)