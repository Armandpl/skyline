from racecar_inference import BUS_PUB_ADDR
import zmq
from mcap_protobuf.writer import Writer
from time import time_ns
from racecar_inference.proto.messages import int_topic_to_message_class
import sys
import logging


# TODO make this a module instead?
def main(fpath):
    # setup mcap writer
    f = open(fpath, "wb")
    mcap_writer = Writer(f)

    # TODO add metadata?
    # should be the configs

    try:
        # setup zmq to connect to the bus
        ctx = zmq.Context()
        sub_socket = ctx.socket(zmq.SUB)
        sub_socket.connect(BUS_PUB_ADDR)

        # sub to every topic
        sub_socket.setsockopt(zmq.SUBSCRIBE, b'')

        while True:
            topic, serialized_message = sub_socket.recv_multipart()
            topic = int.from_bytes(topic, "big")

            if topic != 0 and topic != 1:
                message_type = int_topic_to_message_class[topic]
                message = message_type()
                message.ParseFromString(serialized_message)

                print(f"logger received {message}")

                mcap_writer.write_message(
                    topic=message_type.__name__,
                    message=message,
                    log_time=time_ns(),
                    publish_time=time_ns(),
                )

    except KeyboardInterrupt:
        # on ctrl-c cleanup zmq
        sub_socket.close()
        ctx.term()
        mcap_writer.finish()
        f.close()


# should be run in venv with python>=3.7
if __name__ == "__main__":
    if len(sys.argv) < 2:
        logging.error("pass filepath to write logs to")
    else:
        main(sys.argv[1])