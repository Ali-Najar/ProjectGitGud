import address
import pyMeow as pm
import time

loc = [141.87632751464844,-68.88593292236328,593.8687133789062]

while True:
    pm.w_float(address.process , address.Address.iudex_X , loc[0])
    pm.w_float(address.process , address.Address.iudex_Y , loc[1])
    pm.w_float(address.process , address.Address.iudex_Z , loc[2])