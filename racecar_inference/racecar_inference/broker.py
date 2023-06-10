import multiprocessing
from racecar_inference import BUS_PUB_ADDR, BUS_SUB_ADDR
import logging
import zmq

BROKER_CTRL_ADDR = "ipc:///tmp/broker.ipc"


class Broker(multiprocessing.Process):
    """
    the broker bridges two pub/sub sockets
    this way all processes send to the broket sub
    and all processes sub to the broker pub
    then they can filter messages
    """

    def run(self):
        logging.info("Starting broker")
        context = zmq.Context()

        # Socket to receive messages on
        frontend = context.socket(zmq.XSUB)
        frontend.bind(BUS_SUB_ADDR)

        # Socket to send messages to
        backend = context.socket(zmq.XPUB)
        backend.bind(BUS_PUB_ADDR)

        # listen for TERMINATE signal
        control = context.socket(zmq.SUB)
        control.connect(
            BROKER_CTRL_ADDR
        )  # connect to the address where we'll receive termination signal
        control.setsockopt_string(zmq.SUBSCRIBE, "")

        # forward all messages
        logging.info("Broker running and ready to forward messages")
        zmq.proxy_steerable(frontend=frontend, backend=backend, control=control)

        # cleanup
        logging.info("Broker received terminate, cleaning up")
        frontend.close()
        backend.close()
        control.close()
        context.term()
        logging.info("Broker DONE")
