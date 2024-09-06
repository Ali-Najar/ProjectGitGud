import subprocess
import time
import pydirectinput as pdir
import psutil
import os
import ctypes

import player_xyz as pxyz


def get_process_pid(process_name):   #function to get pid using process name
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'] == process_name:
            return proc.info['pid']
    return None

# Define the number of times you want to run the script
num_runs = 500
process_name = "DarkSoulsIII.exe"

# Use a loop to run the script multiple times
for i in range(num_runs):
    if get_process_pid(process_name) == None:

        game_directory = r"C:\Games\Dark Souls 3\Game"
        executable = r"DarkSoulsIII.exe"
        subprocess.Popen(f'cd /d "{game_directory}" && "{executable}"', shell=True)
        time.sleep(30)
        pdir.press('e')
        time.sleep(4)
        pdir.press('e')
        time.sleep(2)
        pdir.press('e')
        time.sleep(2)
        pdir.press('e')
        time.sleep(2)
        pdir.press('e')
        time.sleep(2)
    time.sleep(10)
    pdir.keyDown('w')
    time.sleep(1)
    pdir.keyUp('w')
    time.sleep(4)
    print("process",i)
    process1 = None
    process2 = None
    process3 = None
    process4 = None
    pdir.keyUp('w')
    pdir.keyUp('s')
    pdir.keyUp('d')
    pdir.keyUp('a')
    pdir.keyUp('o')
    pdir.keyUp('p')
    pdir.keyUp('k')
    pdir.keyUp('j')
    pdir.keyUp('space')
    try:
        pxyz.teleport_to_boss()
        process0 = subprocess.Popen(['python', 'fix_enemy.py'])
        process1 = subprocess.Popen(['python', 'avoid_fall.py'])
        process2 = subprocess.Popen(['python', 'test_cam.py'])
        process3 = subprocess.Popen(['python', 'test_lock.py'])
        process4 = subprocess.Popen(['python', 'test_move.py'])
        process0.terminate()
        subprocess.run(["python", "dodge_training.py"],check=True)
        process1.terminate()
        process2.terminate()
        process3.terminate()
        process4.terminate()
    except:
        if process1 != None:
            process1.terminate()
            process2.terminate()
            process3.terminate()
            process4.terminate()
        pdir.keyDown('w')
        time.sleep(4)
        pdir.keyUp('w')