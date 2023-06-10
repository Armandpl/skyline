from racecar_inference.modules import PidModule, SpeedSensorModule
from racecar_inference.broker import Broker, BROKER_CTRL_ADDR
from racecar_inference import BUS_SUB_ADDR
import logging
import time
import zmq

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

    b = Broker()
    b.start()
    
    pid_module = PidModule("configs/pid.yaml")
    pid_module.start()

    speed_sensor_module = SpeedSensorModule("configs/speed_sensor.yaml")
    speed_sensor_module.start()

    # setup zmq to be able to send commands to the bus
    ctx = zmq.Context()
    pub_socket = ctx.socket(zmq.PUB)
    pub_socket.connect(BUS_SUB_ADDR)

    broker_ctrl = ctx.socket(zmq.PUB)
    broker_ctrl.bind(BROKER_CTRL_ADDR)
    broker_ctrl.setsockopt(zmq.LINGER, 0) # just in case but we shouldn't have > 1 message in the send queue at any time

    def quit_bus():
        # send termination signal to modules
        pub_socket.send_multipart([(0).to_bytes(1, "big"), b''])
        pub_socket.close()

        # send terminate to broker
        broker_ctrl.send_string('TERMINATE')
        broker_ctrl.close()

        # terminate zmq context
        ctx.term()

        # wait for processes to join
        b.join()
        speed_sensor_module.join()
        pid_module.join()

    while True:
        cmd = int(input("what command to send (0: quit, 1: reload):"))
        
        if cmd == 0:
            quit_bus()
            break
        elif cmd == 1:
            pub_socket.send_multipart([(1).to_bytes(1, "big"), b''])

    logging.info("all done")