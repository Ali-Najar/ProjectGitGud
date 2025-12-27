
import pyMeow as pm
import psutil
import pymem
import pymem.process
import time
import pyautogui as pg
import pydirectinput as pdir
import gymnasium 
# import soulsgym
import numpy as np


def get_process_pid(process_name):   #function to get pid using process name
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'] == process_name:
            return proc.info['pid']
    return None

def read_offsets(proc,base_addr,offsets):  #function to add offsets to pointers to get the final address
    ptr = pm.r_int64(proc,base_addr)
    
    for i in offsets[:-1]:
        ptr = pm.r_int64(proc,ptr+i)
    return ptr + offsets[-1]

def print_string(proc , address):
    while pm.r_byte(proc,address)!= 0:
        print(pm.r_string(proc,address),end='')
        address += 2
    print()

def read_string(proc , address):
    str = ''
    while pm.r_byte(proc,address)!= 0:
        str += pm.r_string(proc,address)
        address += 2
    return str


process_name = 'DarkSoulsIII.exe'
process_pid = get_process_pid(process_name)
process = pm.open_process(process_pid)
module = pm.get_module(process, process_name)

class InitPoint:
    iudex = [125.8167495727539 , -64.05046081542969 , 559.1458740234375 , -2.7772343158721924]

class Offset:
    hp = [0x0 , 0x50 , 0x98 , 0x360 , 0x198 , 0x98 , 0xD18]
    estus = [0x38 , 0x230 , 0xE8 , 0x0 , 0x268 , 0x1C0 , 0xB8]
    X = [0x938 , 0xA8 , 0x818 , 0x238 , 0x0 , 0x9F0]
    CamX = [0xD8 , 0x28 , 0x30]
    LockOn = [0x2821]
    player_speed = [0x3A0 , 0x70 , 0xC40]                   #this is the offset from NS_SPRJ::SprjChrPhysicsModule
    player_animation = [0x3A0 , 0x70 , 0x1AD0]              #this is the offset from NS_SPRJ::SprjChrPhysicsModule
    player_animation_name = [0x3A0 , 0x70 , 0xA80]          #this is the offset from NS_SPRJ::SprjChrPhysicsModule
    player_animation_time_old = [0x3A0 , 0x70 , 0x12DC]         #this is the offset from NS_SPRJ::SprjChrPhysicsModule #old version
    player_animation_time = [0x3A0 , 0x70 , 0x12E0]         #this is the offset from NS_SPRJ::SprjChrPhysicsModule
    player_max_animation_time = [0x3A0 , 0x70 , 0x12E4]     #this is the offset from NS_SPRJ::SprjChrPhysicsModule

    iudex_physics = [0x0 , 0x88 , 0x18 , 0x28]
    iudex_hp = [0x0 , 0x88 , 0x18 , 0x28 , 0x3A0 , 0x40 , 0xF8]
    iudex_speed = [0x3A0 , 0x70 , 0xC40]                    #this is the offset from NS_SPRJ::SprjChrPhysicsModule
    iudex_animation = [0x3A0 , 0x70 , 0x1AD0]               #this is the offset from NS_SPRJ::SprjChrPhysicsModule
    iudex_animation_name = [0x3A0 , 0x70 , 0xA80]           #this is the offset from NS_SPRJ::SprjChrPhysicsModule
    iudex_animation_time = [0x3A0 , 0x70 , 0x12E0]          #this is the offset from NS_SPRJ::SprjChrPhysicsModule
    iudex_max_animation_time = [0x3A0 , 0x70 , 0x12E4]      #this is the offset from NS_SPRJ::SprjChrPhysicsModule

    global_speed = [0x0]


class BaseAddress:
    hp = module['base'] + 0x0452CF60
    estus = module['base'] + 0x04644440

    global_speed = module['base'] + 0x0999C28        #Pointer to instance of NS_SPRJ::GlobalSpeed
    no_dead = module['base'] + 0x4768F68        #Pointer to instance of NS_SPRJ::NoDead
    no_hit = module['base'] + 0x4768F72        #Pointer to instance of NS_SPRJ::NoHit
    WorldChrManImp = module['base'] + 0x4768E78 #Pointer to instance of NS_SPRJ::WorldChrManImp
    field_area = module['base'] + 0x04743A80    #Pointer to instance of NS_SPRJ::FieldArea
    LockTgtMan = module['base'] + 0x04766ca0    #Pointer to instance of NS_SPRJ::LockTgtMan
    # GameFlagData = module['base'] + 0x0473be28  #Pointer to instance of NS_SPRJ::GameFlagData
    GameManData = module['base'] + 0x04740178   #Pointer to instance of NS_SPRJ::GameManData

    flag = module['base'] + 0x1404C4590

    iudex = module['base'] + 0x04739958

class Root:
    hp_addr = read_offsets(process,BaseAddress.hp,Offset.hp)
    chr_data_module = hp_addr - 0xD8                                        # address of NS_SPRJ::SprjChrDataModule
    chr_ins_ptr = chr_data_module + 0x8                                     # pointer to NS_SPRJ::PlayerIns
    # print(hex(chr_ins_ptr))
    chr_ins = pm.r_int64(process,chr_ins_ptr)                               # address of NS_SPRJ::PlayerIns
    # print(hex(chr_ins))
    chr_physics_ptr = chr_ins + 0x2428                                      # pointer to NS_SPRJ::SprjChrPhysicsModule
    chr_physics = pm.r_int64(process,chr_physics_ptr)                       # address of NS_SPRJ::SprjChrPhysicsModule
    # print(hex(chr_physics), "FF")
    chr_game_data_ptr = pm.r_int64(process,BaseAddress.GameManData) + 0x10  # Pointer to instance of NS_SPRJ::PlayerGameData

    try:
        iudex_physics_ptr = read_offsets(process,BaseAddress.iudex, Offset.iudex_physics)   # pointer to NS_SPRJ::SprjChrPhysicsModule for iudex
        iudex_physics = pm.r_int64(process,iudex_physics_ptr)                               # address of NS_SPRJ::SprjChrPhysicsModule for iudex
    except:
        print("Iudex failed to load!")


class Address:

    global_speed = BaseAddress.global_speed
    no_dead = BaseAddress.no_dead
    no_hit = BaseAddress.no_hit

    hp = read_offsets(process,BaseAddress.hp,Offset.hp)
    max_hp = hp + 0x4
    max_hp_permanent = max_hp + 0x4
    fp = hp + 0x0C
    max_fp = fp + 0x4
    max_fp_permanent = max_fp + 0x4
    stamina = fp + 0x0C
    max_stamina = stamina + 0x4
    max_stamina_permanent = max_stamina + 0x4

    estus = read_offsets(process,BaseAddress.estus,Offset.estus)

    O = Root.chr_physics + 0x74
    X = Root.chr_physics + 0x80
    Y = Root.chr_physics + 0x84
    Z = Root.chr_physics + 0x88

    player_speed = read_offsets(process,Root.chr_physics_ptr,Offset.player_speed)
    player_animation= read_offsets(process,Root.chr_physics_ptr,Offset.player_animation)
    player_animation_name = read_offsets(process,Root.chr_physics_ptr,Offset.player_animation_name)
    player_animation_time = read_offsets(process,Root.chr_physics_ptr,Offset.player_animation_time)
    player_max_animation_time = read_offsets(process,Root.chr_physics_ptr,Offset.player_max_animation_time)

    try:
        iudex_hp = read_offsets(process,BaseAddress.iudex,Offset.iudex_hp)
        iudex_max_hp = iudex_hp + 0x4
        iudex_O = Root.iudex_physics + 0x74
        iudex_X = Root.iudex_physics + 0x80
        iudex_Y = Root.iudex_physics + 0x84
        iudex_Z = Root.iudex_physics + 0x88

        iudex_speed = read_offsets(process,Root.iudex_physics_ptr,Offset.iudex_speed)
        iudex_animation= read_offsets(process,Root.iudex_physics_ptr,Offset.iudex_animation)
        iudex_animation_name = read_offsets(process,Root.iudex_physics_ptr,Offset.iudex_animation_name)
        iudex_animation_time = read_offsets(process,Root.iudex_physics_ptr,Offset.iudex_animation_time)
        iudex_max_animation_time = read_offsets(process,Root.iudex_physics_ptr,Offset.iudex_max_animation_time)
    except:
        print("Failed to load Iudex addresses!")

    CamX = read_offsets(process ,BaseAddress.field_area,Offset.CamX)
    CamY = CamX + 0x4
    CamZ = CamX + 0x8

    LockOn = read_offsets(process,BaseAddress.LockTgtMan , Offset.LockOn)  # int16


# pm.w_int(process , Address.no_dead, 1)
# while True:
#     x = pm.r_float(process , Address.X)
#     y = pm.r_float(process , Address.Y)
#     z = pm.r_float(process , Address.Z)
#     ex = pm.r_float(process , Address.iudex_X)
#     ey = pm.r_float(process , Address.iudex_Y)
#     ez = pm.r_float(process , Address.iudex_Z)
#     print(x, y, z)
#     print(pm.r_int(process , Address.no_hit))
#     print(np.linalg.norm(np.array([x,y,z]) - np.array([ex,ey,ez])))
# #     print(read_string(process,Address.iudex_animation_name).lower())
#     time.sleep(1)
# print(pm.r_float(process , Address.CamX))
# print(pm.r_float(process , Address.CamY))
# print(pm.r_float(process , Address.CamZ))
# print(pm.w_float(process , BaseAddress.global_speed , 1))
# print(pm.r_int(process , Address.global_speed))
# print(hex(Address.global_speed))
# # print(pm.r_float(process , 0x8B100FF33F800000))
# print(hex(Root.hp_addr))
# print(hex(pm.r_int64(process,Address.global_speed)))
# print(hex(read_offsets(process,BaseAddress.global_speed,Offset.global_speed)))
# print(pm.r_float(process , Address.CamX) , "FD")
# print_string(process , Address.player_animation_name)
# print(hex(Address.O), print(Address.hp), print(Address.CamX), print(Address.Z))

# while True:
#     print(pm.r_int16(process , Address.LockOn))
# # counter = 0
# while True:
# pm.w_int(process, Address.estus , 10)
# pdir.press('r')
# time.sleep(max(0,0.3))
# print(pm.r_float(process,Address.player_max_animation_time))
# pdir.press('r')
# time.sleep(max(0,0.3))
# print(pm.r_float(process,Address.player_max_animation_time))
# pdir.press('r')
# time.sleep(max(0,0.3))
# print(pm.r_float(process,Address.plaWyer_max_animation_time))
# pdir.press('r')
# time.sleep(max(0,0.3))
# print(pm.r_float(process,Address.player_max_animation_time))
# pdir.press('r')
# time.sleep(max(0,0.3))
# print(pm.r_float(process,Address.player_max_animation_time))
            
# def angle_calc(x,y,z,ix,iy,iz,cx,cy,cz):
#         dot_prod = (ix-x)*cx + (iy-y)*cy + (iz-z)*cz
#         dist = np.sqrt((iy-y)**2 + (ix-x)**2 + (iz-z)**2)
#         cam = np.sqrt(cx**2 + cy**2 + cz**2)
#         np.arccos(dot_prod/(cam*dist))
#         return np.arccos(dot_prod/(cam*dist))

# print(hex(Root.iudex_physics_ptr))
# flag = True
# action = None
# xpm.r_int(process , Address.iudex_animation))
# while True:
#     pm.w_int(process , Address.iudex_animation,)
# while True:
#     if action is None:
#         action = read_string(process,Address.iudex_animation_name)
#         print(action)
#     elif action != read_string(process,Address.iudex_animation_name):
#         action = read_string(process,Address.iudex_animation_name)
#         print(pm.r_float(process,Address.iudex_max_animation_time))
#         print(action)
#     # print_string(process,Address.player_animation_name)
#     # print(pm.r_float(process,Address.player_max_animation_time))
#     # time.sleep(max(0,pm.r_float(process,Address.player_max_animation_time)-1.4))
#     # pdir.keyUp('w')
    # if 'roll' not in read_string(process,Address.player_animation_name).lower():
    #     flag = True
    # if 'roll' in read_string(process,Address.player_animation_name).lower() and flag:
    #     time = 0
    #     max_time = 0
    #     if abs(pm.r_float(process,Address.iudex_max_animation_time) - 5) < 0.1:
    #         if abs(pm.r_float(process,Address.iudex_max_animation_time+0x10) - 5) < 0.1:
    #             time = pm.r_float(process , Address.iudex_animation_time+0x20)
    #             max_time = pm.r_float(process , Address.iudex_max_animation_time+0x20)
    #         else:
    #             time = pm.r_float(process , Address.iudex_animation_time+0x10)
    #             max_time = pm.r_float(process , Address.iudex_max_animation_time+0x10)
    #     else:
    #         time = pm.r_float(process , Address.iudex_animation_time)
    #         max_time = pm.r_float(process , Address.iudex_max_animation_time)
    #     ratio = time/max_time# print(read_offsets(process,))
    #     print(ratio)
    #     print(max_time)
    #     print(50*'=')
    #     flag = False
#     x = pm.r_float(process , Address.iudex_X) - pm.r_float(process , Address.X)
#     y = pm.r_float(process , Address.iudex_Y) - pm.r_float(process , Address.Y)
#     z = pm.r_float(process , Address.iudex_Z) - pm.r_float(process , Address.Z)
#     cx = pm.r_float(process , Address.CamX)
#     cy = pm.r_float(process , Address.CamY)
#     cz = pm.r_float(process , Address.CamZ)

#     cam = np.array([cx , cy , cz])
#     second_vec = np.array([x , y , z])

#     dot_prod = np.dot(cam , second_vec)
#     cam_norm = np.linalg.norm(cam)
#     second_vec_norm = np.linalg.norm(second_vec)
#     angle = np.arccos(dot_prod/(cam_norm*second_vec_norm))
    # print(pm.r_float(process , Address.LockOn)) 
    # print(pm.r_int64(process , Address.LockOn)) 
    # print(hex(Address.LockOn))
    # time.sleep(1)
#     pm.w_int(process , Address.hp , 400)



# while True:
#     pm.w_float(process ,Address.CamX ,0.38722047209739685)
#     pm.w_float(process ,Address.CamY ,-0.12185350805521011)
#     pm.w_float(process ,Address.CamZ ,0.9138994216918945)
#     cx = pm.r_float(process, Address.CamX)
#     cy = pm.r_float(process, Address.CamY)
#     cz = pm.r_float(process, Address.CamZ)
#     print(f"CamX:{cx} , CamY:{cy} , CamZ:{cz}")
#     time.sleep(1)
# pm.w_int(process,Address.hp, -1)

#     x = pm.r_float(process, Address.X)
#     y = pm.r_float(process, Address.Y)
#     z = pm.r_float(process, Address.Z)
#     o = pm.r_float(process, Address.O)
#     ix = pm.r_float(process, Address.iudex_X)
#     iy = pm.r_float(process, Address.iudex_Y)
#     iz = pm.r_float(process, Address.iudex_Z)
#     io = pm.r_float(process, Address.iudex_O)
#     print(f"X:{x} , Y:{y} , Z:{z} , O:{o} , iX:{ix} , iY:{iy} , iZ:{iz} , iO:{io}")
#     print(f"CamX:{cx} , CamY:{cy} , CamZ:{cz}")
#     print(f"Angle:{angle_calc(x,y,z,ix,iy,iz,cx,cy,cz)}")
#     for _ in range(20):
#         time.sleep(0.1)
#         pm.w_float(process, Address.iudex_X , 143.87632751464844)
#         pm.w_float(process, Address.iudex_Z , 594.8687133789062)
#         pm.w_float(process, Address.iudex_Y ,-68.88593292236328)
#         pm.w_int(process , Address.hp , 400)
    #     print(read_string(process,Address.player_animation_name))
#     time.sleep(0.5)
#     counter += 0.1
#     pm.w_float(process , Address.O,5)
#     pm.w_float(process , Address.iudex_Y , -60)
#     # if(pm.r_int(process , Address.player_animation) == 4050):
# pm.w_int(process , Address.stamina , 30)

#     pm.w_int(process , Address.hp , 400)
#     time.sleep(0.1)
    # pdir.keyDown('w')
    # pdir.press('space')
    # pdir.keyUp('w')
    # time.sleep(1)
# time.sleep(10)
# pdir.press('q')
# pdir.keyDown('shift')
# pdir.mouseDown(button='right')
# time.sleep(10)
# pdir.keyUp('shift')
# pdir.keyDown('w')
# time.sleep(2)w 
# pdir.keyUp('w')
# pdir.keyDown('a')
# time.sleep(2)
# pdir.keyUp('a')
# pdir.keyDown('s')
# time.sleep(2)
# pdir.keyUp('s')
# print(pg.KEY_NAMES)


