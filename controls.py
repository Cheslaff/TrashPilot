"""
THIS FILE IMPLEMENTS VEHICLE CONTROL FUNCTION.
"""
from vgamepad import VX360Gamepad
import time

# Initialize virtual gamepad
gamepad = VX360Gamepad()


def control_vehicle(steer, acceleration):
    # Reset gamepad controls
    gamepad.left_joystick_float(0.0, 0.0)
    gamepad.right_trigger_float(0.0)
    gamepad.left_trigger_float(0.0)
    gamepad.update()
    time.sleep(0.005)
    # Vehicle contol via gamepad
    if acceleration >= 0:
        gamepad.right_trigger_float(acceleration)
    else:
        gamepad.left_trigger_float(-acceleration)
    gamepad.left_joystick_float(steer, 0.0)
    gamepad.update()
    time.sleep(0.05)
