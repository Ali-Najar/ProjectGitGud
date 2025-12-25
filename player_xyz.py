import pyMeow as pm
import pydirectinput as pdir
import time
import psutil

def get_process_pid(process_name):   #function to get pid using process name
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'] == process_name:
            return proc.info['pid']
    return None




class InitPoint:
    iudex = [125.8167495727539 , -64.05046081542969 , 559.1458740234375 , -2.7772343158721924]


def read_offsets(proc,base_addr,offsets):  #function to add offsets to pointers to get the final address
    ptr = pm.r_int64(proc,base_addr)
    
    for i in offsets[:-1]:
        ptr = pm.r_int64(proc,ptr+i)
    return ptr + offsets[-1]

class Offset:
    hp = [0x0 , 0x50 , 0x98 , 0x360 , 0x198 , 0x98 , 0xD18]
    CamX = [0xD8 , 0x28 , 0x30]

def teleport_to_boss(boss_name="iudex"):
    process_name = 'DarkSoulsIII.exe'
    process_pid = get_process_pid(process_name)
    process = pm.open_process(process_pid)
    module = pm.get_module(process, process_name)
    hp = module['base'] + 0x0452CF60
    field_area = module['base'] + 0x04743A80    #Pointer to instance of NS_SPRJ::FieldArea
    hp_addr = read_offsets(process,hp,Offset.hp)
    chr_data_module = hp_addr - 0xD8                                        # address of NS_SPRJ::SprjChrDataModule
    chr_ins_ptr = chr_data_module + 0x8                                     # pointer to NS_SPRJ::PlayerIns
    chr_ins = pm.r_int64(process,chr_ins_ptr)                               # address of NS_SPRJ::PlayerIns
    chr_physics_ptr = chr_ins + 0x2428                                      # pointer to NS_SPRJ::SprjChrPhysicsModule
    chr_physics = pm.r_int64(process,chr_physics_ptr)  
    O = chr_physics + 0x74
    X = chr_physics + 0x80
    Y = chr_physics + 0x84
    Z = chr_physics + 0x88        # address of NS_SPRJ::SprjChrPhysicsModule
    # CamX = read_offsets(process ,field_area,Offset.CamX)
    # CamY = CamX + 0x4
    # CamZ = CamX + 0x8
    time.sleep(2)
    pm.w_float(process ,X ,InitPoint.iudex[0])
    pm.w_float(process ,Y ,InitPoint.iudex[1])
    pm.w_float(process ,Z ,InitPoint.iudex[2])
    pm.w_float(process ,O ,InitPoint.iudex[3])
    time.sleep(1)
    pdir.press('e')
    time.sleep(3)
    # pdir.keyDown('w')
    # pdir.keyDown('space')
    # time.sleep(5)
    # pdir.keyUp('w')
    pm.w_float(process ,X ,132.74497985839844)
    pm.w_float(process ,Y ,-68.06697082519531)
    pm.w_float(process ,Z ,575.564453125)
    time.sleep(2)
    # print(pm.r_float(process ,X),pm.r_float(process ,Y),pm.r_float(process ,Z))
    # pdir.keyUp('space')
