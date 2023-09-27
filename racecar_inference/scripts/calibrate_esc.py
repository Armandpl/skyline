from racecar_inference.jetracer.nvidia_racecar import NvidiaRacecar

if __name__ == "__main__":
    print("setting up car")
    car = NvidiaRacecar()
    print("car OK")

    car.throttle_gain = 1

    car.throttle = 0.0
    input("throttle 0, press enter to continue")

    car.throttle = 1.0
    input("throttle 1, press enter to continue")

    car.throttle = -1.0
    input("throttle -1, press enter to continue")
    car.throttle = 0
