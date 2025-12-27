import address
import pyMeow as pm
import time
import numpy as np

pm.w_int(address.process, address.Address.estus, 4)
loc = [141.87632751464844,-68.88593292236328,583.8687133789062]
while True:
    if 'fall' in address.read_string(address.process,address.Address.player_animation_name).lower() or pm.r_int(address.process,address.Address.iudex_hp) < 60 or pm.r_int(address.process,address.Address.hp) < 1:
        # pm.w_float(address.process , address.Address.X , loc[0])
        # pm.w_float(address.process , address.Address.Y , loc[1]+2)
        # pm.w_float(address.process , address.Address.Z , loc[2])
        data = np.load('iudex_win_rate.npy')
        win = 0
        if pm.r_int(address.process,address.Address.iudex_hp) < 60:
            win = 1
        # pm.w_int(address.process, address.Address.iudex_hp, 500)
        # pm.w_int(address.process, address.Address.hp, -1)
        data = np.append(data,win)
        np.save('iudex_win_rate',data)
        time.sleep(10)
    
    