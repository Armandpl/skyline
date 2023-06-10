import zmq
import multiprocessing
from omegaconf import OmegaConf
from msgpack import packb, unpackb
import random
import msgpack
import os
import logging
from typing import Optional, Union
from pathlib import Path
import time
from racecar_inference.proto.messages import (
    message_class_to_int_topic,
    message_class_name_to_int_topic,
    int_topic_to_message_class,
)
from racecar_inference.proto.protocol_pb2 import (
    SpeedReading,
    SpeedCommand,
    ThrottleCommand,
)
from racecar_inference import BUS_PUB_ADDR, BUS_SUB_ADDR

# from simple_pid import PID


# TODO set HWM
class BaseModule(multiprocessing.Process):
    def __init__(self, config_path: Optional[Union[Path, str]] = None):
        super().__init__()
        self.config_path = config_path
        self.config = None

    def reload(self):
        """
        these are things we want to do at start and on config reload
        e.g re-instantiate the camera if framerate has changed
        """
        logging.info(f"{self.__class__.__name__} received reload command")

        # NOTE: this assumes configs are dicts and don't have nested params!
        # which is a pretty big assumption
        if self.config is not None:
            prev_conf = OmegaConf.to_container(self.config, resolve=True)
        else:
            prev_conf = None

        self.config = OmegaConf.load(self.config_path)

        # find keys for which value has changed
        if prev_conf is not None:
            changed_keys = []
            new_conf = OmegaConf.to_container(self.config, resolve=True)

            for key in prev_conf.keys() & new_conf.keys():
                if prev_conf[key] != new_conf[key]:
                    changed_keys.append(key)
                    logging.info(
                        f"{self.__class__.__name__} {key}: {dict1[key]} -> {dict2[key]}"
                    )
        else:
            changed_keys = self.config.keys()

        return changed_keys

    def run(self):
        self.reload()
        self.running = True

        # zmq context and socket are not thread safe
        # and for some reason that means we can't them them up in __init__
        self.context = zmq.Context()
        self.pub_socket = self.context.socket(zmq.PUB)
        self.sub_socket = self.context.socket(zmq.SUB)

        self.pub_socket.connect(BUS_SUB_ADDR)
        self.sub_socket.connect(BUS_PUB_ADDR)

        # when terminating discard any unsent message
        self.pub_socket.setsockopt(zmq.LINGER, 0)

        # every module subscribes to quit and reload
        self.subscribe(0)  # quit
        self.subscribe(1)  # reload

        for t in self.config.sub_topics:
            self.subscribe(message_class_name_to_int_topic[t])

        self.loop()

    def loop():
        raise NotImplementedError

    def publish(self, message):
        """
        takes in a protobuf message
        set topic from its class, serialize and send it
        """
        topic = message_class_to_int_topic[message.__class__]
        topic = self.encode_topic(topic)

        serialized_message = (
            message.SerializeToString()
        )  # serialize to string actually outputs bytes
        self.pub_socket.send_multipart([topic, serialized_message])

    def receive(self, noblock=False):
        """
        no block option so that modules that don't need to read from the bus
        can still listen for reload/quit messages
        """
        topic, message = None, None
        try:
            if noblock:
                topic, serialized_message = self.sub_socket.recv_multipart(
                    flags=zmq.NOBLOCK
                )
            else:
                topic, serialized_message = self.sub_socket.recv_multipart()
        except zmq.Again:
            pass

        if topic is not None:
            topic = self.decode_topic(topic)

            # handle special commands
            if topic == 1:
                self.handle_reload()
            elif topic == 0:
                self.handle_quit()
            else:
                message_type = int_topic_to_message_class[topic]
                message = message_type()
                message.ParseFromString(serialized_message)

        return message

    def subscribe(self, topic: int):
        topic = self.encode_topic(topic)
        logging.info(f"{self.__class__.__name__} subscribing to {topic}")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, topic)

    def encode_topic(self, int_topic: int):
        """
        convert topic to single byte to identify messages on bus
        """
        topic = int_topic.to_bytes(length=1, byteorder="big")
        return topic

    def decode_topic(self, byte_topic):
        """
        convert byte topic to int topic
        """
        return int.from_bytes(byte_topic, "big")

    def handle_quit(self):
        logging.info(f"{self.__class__.__name__} received quit command")
        self.running = False  # so that the main loop terminates
        self.pub_socket.close()
        self.sub_socket.close()
        self.context.term()
        logging.info(f"{self.__class__.__name__} DONE")


class SpeedSensorModule(BaseModule):
    def reload(self):
        super().reload()
        # TODO setup serial connection

    def loop(self):
        while self.running:
            # TODO read speed

            # TODO publish speed
            # TODO maybe those topics should be ints
            # this way we're not passing around uncessary bytes
            # ALSO would be cleaner to have constants

            msg = SpeedReading(speed=random.random())
            self.publish(msg)

            # read message in non blocking way
            # in case we receive quit/reload
            # read at the end of the loop else we might send message after receiving quit command
            # TODO this isn't the best best bc we need every module that might publish without receiving
            # to make sure they receive at the end of the loop
            # alternative could be for receive to return a flag and then we break on that
            # but then every module has to implement the break pattern so its the same idk
            self.receive(noblock=True)
            time.sleep(1)


class PidModule(BaseModule):
    def reload(self):
        super().reload()
        # self.pid =

    def loop(self):
        while self.running:
            message = self.receive()

            # if isinstance(message, SpeedCommand):
            #     # update pid command
            # elif isinstance(message, SpeedReading):
            #     # update pid measure
            #     # publish throttle
