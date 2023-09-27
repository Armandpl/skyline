import logging
import time

import zmq

from racecar_inference import BUS_SUB_ADDR
from racecar_inference.broker import BROKER_CTRL_ADDR, Broker
from racecar_inference.modules import (
    CarModule,
    ControlModule,
    PidModule,
    SpeedFilterModule,
    SpeedSensorModule,
)
from racecar_inference.proto.protocol_pb2 import SpeedCommand

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

    modules = []
    modules.append(Broker())
    modules.append(PidModule("configs/pid.yaml"))
    modules.append(SpeedSensorModule("configs/speed_sensor.yaml"))
    modules.append(SpeedFilterModule("configs/speed_filter.yaml"))
    modules.append(CarModule("configs/car.yaml"))
    modules.append(ControlModule("configs/control.yaml"))

    for m in modules:
        m.start()

    # setup zmq to be able to send commands to the bus
    ctx = zmq.Context()
    pub_socket = ctx.socket(zmq.PUB)
    pub_socket.connect(BUS_SUB_ADDR)

    broker_ctrl = ctx.socket(zmq.PUB)
    broker_ctrl.bind(BROKER_CTRL_ADDR)
    broker_ctrl.setsockopt(
        zmq.LINGER, 0  # allow terminating even if we have messages in the send queue
    )  # just in case but we shouldn't have > 1 message in the send queue at any time

    def quit_bus():
        # send termination signal to modules
        pub_socket.send((0).to_bytes(1, "big"))
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
        try:
            # float('a') will crash so make sure we don't crash the control module
            # if we press a key by mistake
            cmd = float(input("what command to send (0: quit, 1: reload"))
        except:
            cmd = -1

        if cmd == 0:
            quit_bus()
            break
        elif cmd == 1:
            pub_socket.send((1).to_bytes(1, "big"))

    logging.info("all done")
