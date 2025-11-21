#!/usr/bin/env python3
# 打印当前 UI 里可控视角（Spectator）的位姿

import time
import carla

def main():
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(5.0)
    world = client.get_world()
    spectator = world.get_spectator()

    # 一次性打印（需要连续打印就用下面的循环版本）
    tr = spectator.get_transform()
    loc, rot = tr.location, tr.rotation
    print(f"Spectator Location: x={loc.x:.3f}, y={loc.y:.3f}, z={loc.z:.3f}")
    print(f"Spectator Rotation: roll={rot.roll:.3f}, pitch={rot.pitch:.3f}, yaw={rot.yaw:.3f}")

if __name__ == "__main__":
    main()
