# spawn_5_vehicles_noview.py
import random
import time
import carla

NUM_VEHICLES = 5
TM_PORT = 8000  # Traffic Manager 默认端口

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)

    # 如果已经是 Town01，可以注释下一行
    world = client.load_world('Town01')

    print("Current map name:", world.get_map().name)
    # 同步模式
    settings = world.get_settings()
    original_settings = settings
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / 20.0
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    tm = client.get_trafficmanager(TM_PORT)
    tm.set_synchronous_mode(True)
    tm.set_global_distance_to_leading_vehicle(2.5)

    blueprint_library = world.get_blueprint_library()
    vehicle_bps = blueprint_library.filter('vehicle.*')

    prefer = ['vehicle.tesla.model3', 'vehicle.audi.tt', 'vehicle.bmw.grandtourer',
              'vehicle.nissan.patrol', 'vehicle.mercedes.coupe']
    preferred_bps = [bp for bp in vehicle_bps if bp.id in prefer]
    if len(preferred_bps) < NUM_VEHICLES:
        preferred_bps = vehicle_bps

    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    actors = []
    try:
        for _ in range(5):
            world.tick()

        count = 0
        i = 0
        while count < NUM_VEHICLES and i < len(spawn_points):
            bp = random.choice(preferred_bps)
            if bp.has_attribute('color'):
                color = random.choice(bp.get_attribute('color').recommended_values)
                bp.set_attribute('color', color)

            transform = spawn_points[i]
            i += 1
            vehicle = world.try_spawn_actor(bp, transform)
            if vehicle is None:
                continue

            vehicle.set_autopilot(True, tm.get_port())
            tm.vehicle_percentage_speed_difference(vehicle, random.randint(-10, 10))

            actors.append(vehicle)
            print(f"Spawned {vehicle.type_id} @ {transform.location} (id={vehicle.id})")
            count += 1
            world.tick()

        print(f"✅ 成功生成 {len(actors)} 辆车，未绑定视角，按 Ctrl+C 退出...")
        while True:
            world.tick()
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n收到退出信号，开始清理...")
    finally:
        for a in actors:
            try:
                a.destroy()
            except:
                pass
        world.apply_settings(original_settings)
        tm.set_synchronous_mode(False)
        print("已清理并恢复设置。")

if __name__ == "__main__":
    main()
