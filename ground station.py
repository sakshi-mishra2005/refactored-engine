import cv2
import numpy as np
import zmq
import argparse
import time
 
 
def send_target_image(socket, image_path: str):
    """Reads an image file and sends it to the drone as a target."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"[GS] Could not load image: {image_path}")
        return
    _, buf    = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    payload   = b"TARGET:" + buf.tobytes()
    socket.send(payload)
    print(f"[GS] Target sent: {image_path}  ({len(payload)//1024} KB)")
 
 
def main(pi_ip: str, zmq_port: int, initial_target: str):
    print(f"[GS] Connecting to drone at {pi_ip}...")
 
    ctx = zmq.Context()
 
    # receive video stream from drone
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://{pi_ip}:{zmq_port}")
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.setsockopt(zmq.RCVTIMEO, 3000)
 
    # send target images to drone
    pub = ctx.socket(zmq.PUB)
    pub.connect(f"tcp://{pi_ip}:{zmq_port + 1}")
    time.sleep(0.5)  # give ZMQ time to connect
 
    # send initial target if provided
    if initial_target:
        send_target_image(pub, initial_target)
 
    print("[GS] Live feed started.")
    print("  T = send new target image")
    print("  Q = quit viewer\n")
 
    frame_count = 0
    fps_timer   = time.time()
    fps         = 0.0
 
    while True:
        try:
            raw   = sub.recv()
            buf   = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
 
            if frame is None:
                continue
 
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_timer)
                fps_timer = time.time()
 
            cv2.putText(frame, f"FPS: {fps:.1f}  |  T=new target  Q=quit",
                (6, frame.shape[0] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
 
            cv2.imshow("Drone Ground Station", frame)
 
        except zmq.Again:
            print("[GS] No frame received — is the drone script running?")
            # show blank frame so window stays open
            blank = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for drone...",
                (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
            cv2.imshow("Drone Ground Station", blank)
 
        key = cv2.waitKey(1) & 0xFF
 
        if key in (ord('q'), ord('Q'), 27):
            print("[GS] Quit.")
            break
 
        elif key in (ord('t'), ord('T')):
            path = input("Enter path to target image: ").strip()
            if path:
                send_target_image(pub, path)
 
    cv2.destroyAllWindows()
    sub.close()
    pub.close()
    ctx.term()
 
 
if _name_ == "_main_":
    parser = argparse.ArgumentParser(description="Drone ground station viewer")
    parser.add_argument("--ip",     type=str, required=True,
                        help="Raspberry Pi IP address  e.g. 192.168.1.42")
    parser.add_argument("--port",   type=int, default=5555,
                        help="ZMQ base port (default 5555)")
    parser.add_argument("--target", type=str, default="",
                        help="Target image to send on startup (optional)")
    args = parser.parse_args()
    main(args.ip, args.port, args.target)