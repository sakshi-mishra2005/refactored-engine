hey this is sakshi trying to learn git 
drone controller .py added
"""
drone_controller.py  —  Run this ON your Raspberry Pi
======================================================
Hardware assumed:
  - Raspberry Pi 4 (any)
  - Pi Camera Module (CSI ribbon cable)
  - Pixhawk flight controller (USB or UART to Pi)
  - LiDAR sensor (USB or I2C to Pi)
  - PX4FLOW optical flow sensor (plugged into Pixhawk, NOT Pi)
  - 4x Brushless motors + ESCs on drone frame
  - LiPo battery

Install on Pi:
  pip install dronekit pymavlink opencv-python numpy pyzmq rplidar-roboticia --break-system-packages
  sudo apt install -y python3-picamera2

What this code does:
  1. Connects to Pixhawk via MAVLink (DroneKit)
  2. Arms and takes off to a target altitude
  3. Hovers stably using optical flow data from Pixhawk
  4. Watches camera for a target image sent from laptop
  5. Flies toward the target when found
  6. Uses LiDAR for obstacle avoidance
  7. Streams live video back to laptop
  8. Lands safely when target is reached or battery is low

Usage:
  python3 drone_controller.py --target target.jpg --altitude 2.0
"""

ground_station.py  —  Run this ON your Laptop
==============================================
What this does:
  1. Opens a live video window showing drone's camera feed
  2. Lets you send a target image to the drone by pressing T
  3. Shows telemetry overlay (altitude, battery, obstacle distance)
  4. Press Q to quit viewer (drone keeps flying)
 
Install on laptop:
  pip install opencv-python numpy pyzmq
 
Usage:
  python3 ground_station.py --ip 192.168.1.42
  python3 ground_station.py --ip 192.168.1.42 --target my_object.jpg
 
  Find Pi IP:  run  hostname -I  on the Pi terminal


calling the files 
  python3 drone_controller.py --target target.jpg --altitude 2.0

  python3 ground_station.py --ip 192.168.1.42 --target my_object.jpg
