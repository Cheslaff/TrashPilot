"""
THIS CODE IMPLEMENTS DATA COLLECTION.
"""
import time
import mss
import pygetwindow as gw
from PIL import Image
import time
from pygame import mixer  # Load the popular external library
import pygame

# Countdown
for i in reversed(range(5)):
    print(i+1)
    time.sleep(1)

# Get the game window by title
window_title = "BeamNG.drive - 0.36.4.0.18364 - RELEASE - Direct3D11"
game_win = gw.getWindowsWithTitle(window_title)

if not game_win:
    raise Exception(f"No window with title '{window_title}' found.")

game_win = game_win[0]

if game_win.isMinimized or not game_win.isActive:
    print("Warning: Window is minimized or not active.")

# variables
left, top, width, height = game_win.left, game_win.top, game_win.width, game_win.height
fps = 15
frame_duration = 1 / fps

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    raise RuntimeError("No gamepad connected!")

joystick = pygame.joystick.Joystick(0)
joystick.init()

print(f"Gamepad name: {joystick.get_name()}")
print(f"Axes available: {joystick.get_numaxes()}")

steering_axis = 0  # steering
br_axis = 4  # braking
tr_axis = 5  # acceleration


# Create an MSS instance
with mss.mss() as sct:
    monitor = {"top": top, "left": left, "width": width, "height": height}
    
    for i in range(15*60*40):  # 40 minutes
        start = time.time()

        img = sct.grab(monitor)
        img_pil = Image.frombytes("RGB", img.size, img.rgb)
        img_pil = img_pil.resize((320, 240))
        start = time.time()

        pygame.event.pump()
        steering = joystick.get_axis(steering_axis)
        throttle = joystick.get_axis(tr_axis)
        brake = joystick.get_axis(br_axis)

        acc = (throttle + 1) / 2 if throttle >= brake else -(brake + 1) / 2
        print(f"🛞{steering:.2f}, 🔥{acc:.2f}")

        img_pil.save(f"data/frame_{i:03}_{steering}_{acc}.png")

        time.sleep(max(0, frame_duration - (time.time() - start)))
