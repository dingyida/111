import carla
client = carla.Client('localhost', 2000)
print(client.get_world().get_map().name)
