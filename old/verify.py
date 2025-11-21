import math


def euler_to_vector(pitch_deg, yaw_deg):
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    x = math.cos(pitch) * math.cos(yaw)
    y = math.cos(pitch) * math.sin(yaw)
    z = math.sin(pitch)

    return (x, y, z)

euler_to_vector(0.0, 1.4899946451187134)
print(euler_to_vector(8.42, 110.46))