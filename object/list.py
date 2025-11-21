import carla
client = carla.Client('localhost', 2000)
client.set_timeout(5.0)
world = client.get_world()
for bp in world.get_blueprint_library().filter('*trash*can*'):
    print(bp.id)

