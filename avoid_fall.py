import address
import pyMeow as pm
import time

loc = [141.87632751464844,-68.88593292236328,583.8687133789062]
while True:
    if 'fall' in address.read_string(address.process,address.Address.player_animation_name).lower():
        pm.w_float(address.process , address.Address.X , loc[0])
        pm.w_float(address.process , address.Address.Y , loc[1]+2)
        pm.w_float(address.process , address.Address.Z , loc[2])
        time.sleep(2)
    