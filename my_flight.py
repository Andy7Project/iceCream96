"""
Roomba style autonomous drone for AirSim (Unreal Engine)

This implemntation does not rely on the drone position, map, GPS or SLAM. It only uses:
  * distance sensors  (front, front-left, front-right, left, right) and camera -> "whiskers"
  * the collision flag  -> "bump sensor"
  * a fixed altitude setpoint  -> "keep at floor height"
  * a timer and a random number generator -> "Roomba brain"

Behaviours are all of the drones different modes (like a Roomba):
  SPIRAL       - expanding spiral to cover the area around the start (does entire 360 to target balloon)
  CRUISE       - go straight until something is ahead
  TURN         - rotate a random angle away from the obstacle
  WALL_FOLLOW  - hug the wall on the right side for a random time (for random cone shapes)
  ESCAPE       - after a collision: back up, then turn randomly

Requires: you're gonna need to pip install airsim numpy
Put the sensors from the settings.json snippet at the bottom in
~/Documents/AirSim/settings.json, then start the Unreal map and run this file.
"""

import math
import random
import time
from enum import Enum, auto

import airsim


# Tunables

CFG = dict(
    vehicle_name="",
    altitude_m=2.0,          # hover height above takeoff point (barometric hold, not a map position)
    loop_hz=10.0,

    cruise_speed=2.0,        # m/s forward
    spiral_speed=1.5,
    wall_speed=1.2,
    escape_speed=1.0,

    front_stop_dist=2.5,     # m: something ahead -> react
    front_slow_dist=5.0,     # m: start slowing down
    side_clear_dist=1.5,     # m: min allowed side clearance when cruising
    wall_target_dist=2.0,    # m: distance to keep from wall while following
    wall_lost_dist=6.0,      # m: wall considered gone (outer corner)

    turn_rate_dps=60.0,      # deg/s yaw rate while turning
    turn_min_deg=60.0,
    turn_max_deg=180.0,

    spiral_time=(15.0, 30.0),   # s, random range
    cruise_time_max=40.0,       # s, then do something different
    wall_time=(15.0, 40.0),     # s, random range
    wall_follow_prob=0.45,      # chance to wall-follow after an obstacle (else just turn)
    spiral_r0=0.6,              # m, starting radius of spiral
    spiral_growth=0.25,         # m of radius gained per second

    sensor_max=20.0,
)

SENSORS = ["DistFront", "DistFrontLeft", "DistFrontRight", "DistLeft", "DistRight"]


class State(Enum):
    SPIRAL = auto()
    CRUISE = auto()
    TURN = auto()
    WALL_FOLLOW = auto()
    ESCAPE = auto()



# Sense

class Senses:
    def __init__(self, client, vehicle):
        self.c = client
        self.v = vehicle
        self.last_collision_ts = 0

    def distances(self):
        d = {}
        for name in SENSORS:
            try:
                r = self.c.getDistanceSensorData(name, self.v).distance
                d[name] = min(max(r, 0.0), CFG["sensor_max"])
            except Exception:
                d[name] = CFG["sensor_max"]
        return d

    def bumped(self):
        """True only once per new collision event."""
        info = self.c.simGetCollisionInfo(self.v)
        if info.has_collided and info.time_stamp != self.last_collision_ts:
            self.last_collision_ts = info.time_stamp
            return True
        return False


# Actuation (all commands are in the BODY frame - no world coordinates)

class Actuator:
    def __init__(self, client, vehicle, z_ned):
        self.c = client
        self.v = vehicle
        self.z = z_ned
        self.dt = 1.0 / CFG["loop_hz"]

    def move(self, forward=0.0, right=0.0, yaw_rate_dps=0.0):
        self.c.moveByVelocityZBodyFrameAsync(
            forward, right, self.z, self.dt * 1.5,
            airsim.DrivetrainType.MaxDegreeOfFreedom,
            airsim.YawMode(True, yaw_rate_dps),
            vehicle_name=self.v,
        )

    def stop(self):
        self.move(0, 0, 0)


# The brain

class RoombaBrain:
    def __init__(self, senses, act):
        self.s = senses
        self.a = act
        self.state = None
        self.t_enter = 0.0
        self.t_limit = 0.0
        self.turn_dir = 1          # +1 = turn right (clockwise), -1 = left
        self.after_turn = State.CRUISE
        self.after_escape_turn_deg = 0.0
        self.escape_phase = "back"
        self.enter(State.SPIRAL)

    # finite state helpers
    def elapsed(self):
        return time.time() - self.t_enter

    def enter(self, new_state, **kw):
        self.state = new_state
        self.t_enter = time.time()
        if new_state == State.SPIRAL:
            self.t_limit = random.uniform(*CFG["spiral_time"])
        elif new_state == State.CRUISE:
            self.t_limit = CFG["cruise_time_max"]
        elif new_state == State.WALL_FOLLOW:
            self.t_limit = random.uniform(*CFG["wall_time"])
        elif new_state == State.TURN:
            deg = kw.get("deg", random.uniform(CFG["turn_min_deg"], CFG["turn_max_deg"]))
            self.turn_dir = kw.get("direction", random.choice([-1, 1]))
            self.t_limit = deg / CFG["turn_rate_dps"]
            self.after_turn = kw.get("then", State.CRUISE)
        elif new_state == State.ESCAPE:
            self.escape_phase = "back"
            self.t_limit = 1.2
        print(f"[brain] -> {new_state.name} (limit {self.t_limit:.1f}s)")

    @staticmethod
    def more_open_side(d):
        left = min(d["DistLeft"], d["DistFrontLeft"])
        right = min(d["DistRight"], d["DistFrontRight"])
        return 1 if right > left else -1   # +1 = turn right

    def obstacle_ahead(self, d):
        return min(d["DistFront"], d["DistFrontLeft"], d["DistFrontRight"]) < CFG["front_stop_dist"]

    # control tick / loop
    def step(self):
        d = self.s.distances()

        # Bump sensor overrides everything
        if self.s.bumped() and self.state != State.ESCAPE:
            self.enter(State.ESCAPE)

        handler = {
            State.SPIRAL: self.do_spiral,
            State.CRUISE: self.do_cruise,
            State.TURN: self.do_turn,
            State.WALL_FOLLOW: self.do_wall_follow,
            State.ESCAPE: self.do_escape,
        }[self.state]
        handler(d)

    #  behaviours 
    def do_spiral(self, d):
        if self.obstacle_ahead(d):
            self.on_obstacle(d)
            return
        if self.elapsed() > self.t_limit:
            self.enter(State.CRUISE)
            return
        v = CFG["spiral_speed"]
        r = CFG["spiral_r0"] + CFG["spiral_growth"] * self.elapsed()
        yaw_rate = math.degrees(v / r)               # omega = v / r
        yaw_rate = min(yaw_rate, 90.0)
        self.a.move(forward=v, yaw_rate_dps=yaw_rate)

    def do_cruise(self, d):
        front = min(d["DistFront"], d["DistFrontLeft"], d["DistFrontRight"])
        if front < CFG["front_stop_dist"]:
            self.on_obstacle(d)
            return
        if self.elapsed() > self.t_limit:
            self.enter(State.SPIRAL)
            return

        # slow down as things get close
        speed = CFG["cruise_speed"]
        if front < CFG["front_slow_dist"]:
            frac = (front - CFG["front_stop_dist"]) / (CFG["front_slow_dist"] - CFG["front_stop_dist"])
            speed *= max(0.3, frac)

        # gentle side avoidance so we don't scrape walls while cruising
        lateral, yaw = 0.0, 0.0
        if d["DistLeft"] < CFG["side_clear_dist"]:
            lateral, yaw = 0.6, 10.0
        elif d["DistRight"] < CFG["side_clear_dist"]:
            lateral, yaw = -0.6, -10.0
        self.a.move(forward=speed, right=lateral, yaw_rate_dps=yaw)

    def on_obstacle(self, d):
        """Roomba choice: sometimes hug the wall, otherwise bounce off at a random angle."""
        side = self.more_open_side(d)
        if random.random() < CFG["wall_follow_prob"]:
            # turn LEFT (away from a right-hand wall), then follow it with the right side
            self.enter(State.TURN, direction=-1, deg=random.uniform(60, 100), then=State.WALL_FOLLOW)
        else:
            self.enter(State.TURN, direction=side, then=State.CRUISE)

    def do_turn(self, d):
        if self.elapsed() >= self.t_limit:
            self.enter(self.after_turn)
            return
        self.a.move(forward=0.0, yaw_rate_dps=self.turn_dir * CFG["turn_rate_dps"])

    def do_wall_follow(self, d):
        """Keep the wall on the RIGHT at wall_target_dist using a P-controller."""
        front = min(d["DistFront"], d["DistFrontLeft"], d["DistFrontRight"])
        right = min(d["DistRight"], d["DistFrontRight"] * 0.8)   # front-right sees corners early

        if self.elapsed() > self.t_limit:
            self.enter(State.SPIRAL)
            return

        # inside corner: wall ahead -> rotate left in place
        if front < CFG["front_stop_dist"]:
            self.a.move(forward=0.0, yaw_rate_dps=-CFG["turn_rate_dps"])
            return

        # outer corner / lost the wall: arc right to find it again
        if d["DistRight"] > CFG["wall_lost_dist"]:
            self.a.move(forward=CFG["wall_speed"] * 0.6, yaw_rate_dps=35.0)
            return

        error = right - CFG["wall_target_dist"]            # >0 too far, <0 too close
        lateral = max(-1.0, min(1.0, 0.5 * error))         # +right
        yaw = max(-25.0, min(25.0, 12.0 * error))          # turn toward wall if far
        self.a.move(forward=CFG["wall_speed"], right=lateral, yaw_rate_dps=yaw)

    def do_escape(self, d):
        if self.escape_phase == "back":
            # back away from whatever we hit, preferring the open direction
            self.a.move(forward=-CFG["escape_speed"])
            if self.elapsed() > self.t_limit:
                self.enter(State.TURN, direction=self.more_open_side(d), deg=random.uniform(90, 180),
                           then=State.CRUISE)


# Main

def main():
    vehicle = CFG["vehicle_name"]
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True, vehicle)
    client.armDisarm(True, vehicle)

    print("Taking off...")
    client.takeoffAsync(vehicle_name=vehicle).join()

    # Altitude setpoint relative to where we started (NED: negative = up).
    # This is a height hold only; the drone still has no idea where it is horizontally.
    z_ned = -CFG["altitude_m"]
    client.moveToZAsync(z_ned, 1.5, vehicle_name=vehicle).join()

    senses = Senses(client, vehicle)
    act = Actuator(client, vehicle, z_ned)
    brain = RoombaBrain(senses, act)

    period = 1.0 / CFG["loop_hz"]
    try:
        while True:
            t0 = time.time()
            brain.step()
            time.sleep(max(0.0, period - (time.time() - t0)))
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        act.stop()
        client.hoverAsync(vehicle_name=vehicle).join()
        client.landAsync(vehicle_name=vehicle).join()
        client.armDisarm(False, vehicle)
        client.enableApiControl(False, vehicle)


if __name__ == "__main__":
    main()

    #IMPORT THIS INTO AIRSIM FOR THE CORRECT FUNCTIONALITY 
"""
settings.json  (~/Documents/AirSim/settings.json)  - adds the 5 "whisker" sensors
Yaw is relative to the drone's nose. SensorType 5 = Distance.
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "Vehicles": {
    "SimpleFlight": {
      "VehicleType": "SimpleFlight",
      "Sensors": {
        "DistFront":      {"SensorType": 5, "Enabled": true, "MinDistance": 0.2, "MaxDistance": 20, "X": 0.3, "Y": 0,   "Z": 0, "Yaw": 0,   "Pitch": 0, "Roll": 0},
        "DistFrontLeft":  {"SensorType": 5, "Enabled": true, "MinDistance": 0.2, "MaxDistance": 20, "X": 0.3, "Y": -0.2,"Z": 0, "Yaw": -40, "Pitch": 0, "Roll": 0},
        "DistFrontRight": {"SensorType": 5, "Enabled": true, "MinDistance": 0.2, "MaxDistance": 20, "X": 0.3, "Y": 0.2, "Z": 0, "Yaw": 40,  "Pitch": 0, "Roll": 0},
        "DistLeft":       {"SensorType": 5, "Enabled": true, "MinDistance": 0.2, "MaxDistance": 20, "X": 0,   "Y": -0.3,"Z": 0, "Yaw": -90, "Pitch": 0, "Roll": 0},
        "DistRight":      {"SensorType": 5, "Enabled": true, "MinDistance": 0.2, "MaxDistance": 20, "X": 0,   "Y": 0.3, "Z": 0, "Yaw": 90,  "Pitch": 0, "Roll": 0}
      }
    }
  }
}
"""
