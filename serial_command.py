import serial
import time
import numpy as np

class SerialCommander:
    def __init__(self, port='/dev/ttyUSB0', baud=9600, min_angle_change=1.0):
        self.ser        = serial.Serial(port, baud, timeout=1)
        self.last_angle = None
        self.min_change = min_angle_change
        time.sleep(2)  # wait for Arduino reset
        print(f"Serial connected on {port} at {baud} baud")

    def send_angle(self, angle_deg):
        """Send only if angle changed by more than min_change degrees."""
        angle_deg = int(np.clip(angle_deg, -90, 90))
        if self.last_angle is None or abs(angle_deg - self.last_angle) >= self.min_change:
            cmd = f"A{angle_deg:03d}\n"
            self.ser.write(cmd.encode())
            self.last_angle = angle_deg
            return True   # actually sent
        return False      # skipped

    def read_response(self):
        if self.ser.in_waiting:
            print(self.ser.readline().decode().strip())

    def close(self):
        self.ser.close()
        print("Serial closed")


def compute_tilt_angle(box, frame_h, vfov_deg, camera_tilt_deg):
    """
    Compute servo tilt angle to point at detected box.

    Args:
        box            : (x1, y1, x2, y2)
        frame_h        : frame height in pixels
        vfov_deg       : camera vertical field of view in degrees
        camera_tilt_deg: physical tilt of camera from horizontal
    """
    x1, y1, x2, y2 = box
    cy           = (y1 + y2) / 2
    deg_per_px   = vfov_deg / frame_h
    pixel_offset = (frame_h / 2) - cy      # positive = drone above centre
    angle_offset = pixel_offset * deg_per_px
    tilt         = camera_tilt_deg + angle_offset
    # print(f"cy={cy:.0f}  frame_centre={frame_h/2}  pixel_offset={pixel_offset:.1f} tilt = {tilt:.1f}")
    return float(np.clip(tilt, -90, 90))

