
import cosysairsim as airsim
import os
import time

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

client.takeoffAsync().join()

os.makedirs("temp", exist_ok=True)

while True:
    # Read distance sensor
    distance_data = client.getDistanceSensorData(
        distance_sensor_name="distance",
        vehicle_name="airsimvehicle"
    )
    print(f"Closest object distance: {distance_data.distance:.2f} m")

    # Capture image from camera 0
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene)
    ])

    print(f"Retrieved images: {len(responses)}")

    # Save captured images
    for response in responses:
        if response.width > 0 and response.height > 0:
            airsim.write_file(
                os.path.normpath("temp/py1.png"),
                response.image_data_uint8
            )

    time.sleep(0.1)