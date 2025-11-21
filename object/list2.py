import carla

# Connect to the CARLA server (make sure CarlaUE4.exe is running)
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# Get the current world
world = client.get_world()

# Get the blueprint library
blueprints = world.get_blueprint_library()

# Print all blueprint IDs
print("=== Blueprint List ===")
for bp in blueprints:
    print(bp.id)
print(f"Total blueprints: {len(blueprints)}")
