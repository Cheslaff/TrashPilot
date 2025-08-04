"""
THIS FILE RUNS A GAME CONTOL LOOP WITH A PRETRAINED MODEL.
"""
import time
import mss
import pygetwindow as gw
from PIL import Image
import time
from pygame import mixer  # Load the popular external library
import torch
from main import MobileNetV2
from torchvision.transforms import ToTensor
from controls import control_vehicle

# Model Loading
model = MobileNetV2(in_channels=9).to("cuda").eval()
model.load_state_dict(torch.load("archive(V0.5)/model_params.pth", weights_only=True))

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

# vaiables
left, top, width, height = game_win.left, game_win.top, game_win.width, game_win.height
fps = 15
frame_duration = 1 / fps

with mss.mss() as sct:
    monitor = {"top": top, "left": left, "width": width, "height": height}
    
    for i in range(15*60*5):
        start = time.time()

        img = sct.grab(monitor)
        img_pil = Image.frombytes("RGB", img.size, img.rgb).resize((320, 240))
        start = time.time()

        img_pil.save(f"sensor_data/frame_{i:03}.png")

        if i >= 2:
            frame1 = ToTensor()(Image.open(f"sensor_data/frame_{i-2:03}.png").convert("RGB"))
            frame2 = ToTensor()(Image.open(f"sensor_data/frame_{i-1:03}.png").convert("RGB"))
            frame3 = ToTensor()(Image.open(f"sensor_data/frame_{i:03}.png").convert("RGB"))
            final = torch.concat([frame1, frame2, frame3], dim=0).unsqueeze(0).to("cuda")
            pred = torch.tanh(model(final))
            steering = pred[0][0].item()
            steering += 0.15
            acc = pred[0][1].item()
            
            control_vehicle(steering, acc*0.7)

        time.sleep(max(0, frame_duration - (time.time() - start)))
