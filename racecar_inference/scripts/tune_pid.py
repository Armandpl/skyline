import logging
import time

import zmq

from racecar_inference import BUS_SUB_ADDR
from racecar_inference.broker import BROKER_CTRL_ADDR, Broker
from racecar_inference.modules import CarModule, PidModule, SpeedSensorModule
from racecar_inference.proto.protocol_pb2 import SpeedCommand

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

    modules = []
    modules.append(Broker())
    modules.append(PidModule("configs/pid.yaml"))
    modules.append(SpeedSensorModule("configs/speed_sensor.yaml"))
    modules.append(CarModule("configs/car.yaml"))

    for m in modules:
        m.start()

    # setup zmq to be able to send commands to the bus
    ctx = zmq.Context()
    pub_socket = ctx.socket(zmq.PUB)
    pub_socket.connect(BUS_SUB_ADDR)

    broker_ctrl = ctx.socket(zmq.PUB)
    broker_ctrl.bind(BROKER_CTRL_ADDR)
    broker_ctrl.setsockopt(
        zmq.LINGER, 0
    )  # just in case but we shouldn't have > 1 message in the send queue at any time

    def quit_bus():
        # send termination signal to modules
        pub_socket.send_multipart([(0).to_bytes(1, "big"), b""])
        pub_socket.close()

        # send terminate to broker
        broker_ctrl.send_string("TERMINATE")
        broker_ctrl.close()

        # terminate zmq context
        ctx.term()

        # wait for processes to join
        for m in modules:
            m.join()

    while True:
        cmd = float(
            input("what command to send (0: quit, 1: reload, >1 speed command):")
        )

        if cmd == 0:
            quit_bus()
            break
        elif cmd == 1:
            pub_socket.send_multipart([(1).to_bytes(1, "big"), b""])
        elif (
            cmd > 1 and cmd < 4
        ):  # make sure we don't send a launch command and send the car to mars
            msg = SpeedCommand(desired_speed=cmd)
            cmd_byte = 3
            pub_socket.send_multipart(
                [(cmd_byte).to_bytes(1, "big"), msg.SerializeToString()]
            )

    logging.info("all done")
