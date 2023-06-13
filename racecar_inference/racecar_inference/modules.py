import logging
import multiprocessing
import os
import random
import time
from pathlib import Path
from typing import Optional, Union

import cv2
import msgpack
import numpy as np
import serial
import torch
import zmq
from msgpack import packb, unpackb
from omegaconf import OmegaConf
from simple_pid import PID
from torch2trt import TRTModule
from torchvision import transforms

from racecar_inference import BUS_PUB_ADDR, BUS_SUB_ADDR
from racecar_inference.jetcam.csi_camera import CSICamera
from racecar_inference.jetracer.nvidia_racecar import NvidiaRacecar
from racecar_inference.proto.messages import (
    int_topic_to_message_class,
    message_class_name_to_int_topic,
    message_class_to_int_topic,
)
from racecar_inference.proto.protocol_pb2 import (
    SpeedCommand,
    SpeedReading,
    SteeringCommand,
    ThrottleCommand,
)


# TODO set HWM
class BaseModule(multiprocessing.Process):
    def __init__(self, config_path: Optional[Union[Path, str]] = None):
        super().__init__()
        self.config_path = config_path
        self.cfg = None

    def init(self):
        """
        runs once at the start of the run method
        we do this instead of __init__ bc some object are not thread safe
        TODO check if we really cant use __init__
        """
        pass

    def reload(self):
        """
        these are things we want to do at start and on config reload
        e.g re-instantiate the camera if framerate has changed
        """
        self.log("info", "RELOADING")

        # NOTE: this assumes configs are dicts and don't have nested params!
        # which is a pretty big assumption
        if self.cfg is not None:
            prev_conf = OmegaConf.to_container(self.cfg, resolve=True)
        else:
            prev_conf = None

        self.cfg = OmegaConf.load(self.config_path)

        # find keys for which value has changed
        if prev_conf is not None:
            changed_keys = []
            new_conf = OmegaConf.to_container(self.cfg, resolve=True)

            for key in prev_conf.keys() & new_conf.keys():
                if prev_conf[key] != new_conf[key]:
                    changed_keys.append(key)
                    self.log(
                        "info",
                        f"{key}: {prev_conf[key]} -> {new_conf[key]}"
                    )
        else:
            changed_keys = self.cfg.keys()

        return changed_keys

    def run(self):
        self.init()
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

        for t in self.cfg.sub_topics:
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
                self.reload()
            elif topic == 0:
                self.handle_quit()
            else:
                message_type = int_topic_to_message_class[topic]
                message = message_type()
                message.ParseFromString(serialized_message)

        return message

    def subscribe(self, topic: int):
        try:
            msg_type = int_topic_to_message_class[topic].__name__
        except AttributeError:
            msg_type = topic

        topic = self.encode_topic(topic)
        self.log("info", f"subscribing to {msg_type}")
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
        self.log("info", "received quit command")
        self.running = False  # so that the main loop terminates
        self.pub_socket.close()
        self.sub_socket.close()
        self.context.term()
        self.log("info", "DONE")

    def log(self, level, s):
        if level not in ["debug", "info", "warning", "error", "critical"]:
            raise ValueError(
                "Invalid log level. Must be one of 'debug', 'info', 'warning', 'error', or 'critical'."
            )
        getattr(logging, level)(f"{self.__class__.__name__}: {s}")


class SpeedSensorModule(BaseModule):
    def init(self):
        self.ser = None

    def reload(self):
        changed_keys = super().reload()
        if (
            "ser_device" in changed_keys
            or "ser_baudrate" in changed_keys
            or "ser_timeout" in changed_keys
        ):
            if self.ser is not None:
                self.ser.close()

            self.log("info", "Setting up serial device")
            self.ser = serial.Serial(
                self.cfg.ser_device, self.cfg.ser_baudrate, timeout=self.cfg.ser_timeout
            )

    def handle_quit(self):
        super().handle_quit()
        self.ser.close()

    def loop(self):
        while self.running:
            line = self.ser.readline()  # read a '\n' terminated line
            speed = float(line.decode("utf-8"))

            msg = SpeedReading(speed=speed)
            self.publish(msg)

            # read message in non blocking way
            # in case we receive quit/reload
            # read at the end of the loop else we might send message after receiving quit command
            # TODO this isn't the best best bc we need every module that might publish without receiving
            # to make sure they receive at the end of the loop
            # alternative could be for receive to return a flag and then we break on that
            # but then every module has to implement the break pattern so its the same idk
            self.receive(noblock=True)


class PidModule(BaseModule):
    def reload(self):
        changed_keys = super().reload()
        if len(changed_keys) > 0:
            # TODO set dt for pid? understand how its used and if its needed
            self.pid_controller = PID(
                Kp=self.cfg.p,
                Ki=self.cfg.i,
                Kd=self.cfg.d,
                setpoint=0.0,
                output_limits=(self.cfg.min_throttle, self.cfg.max_throttle),
            )

    def loop(self):
        while self.running:
            message = self.receive()

            # TODO implement failsafe logic if its been a while seen we received a speed update
            # well actually we update the pid when we receive a speed reading
            # so worst case throttle stays fixed at last value

            if isinstance(message, SpeedCommand):
                # update pid command
                target_speed = message.desired_speed
                self.pid_controller.setpoint = target_speed
            elif isinstance(message, SpeedReading):
                throttle = self.pid_controller(message.speed)
                self.publish(ThrottleCommand(throttle=throttle))


# TODO handle quit del car
class CarModule(BaseModule):
    def init(self):
        self.car = None

    def reload(self):
        changed_keys = super().reload()

        if self.car is None:
            self.car = NvidiaRacecar()

        if len(changed_keys) > 0:
            self.car.throttle_gain = self.cfg.throttle_gain
            self.car.steering_gain = self.cfg.steering_gain
            self.car.steering_offset = self.cfg.steering_offset

    def loop(self):
        while self.running:
            message = self.receive()

            if isinstance(message, ThrottleCommand):
                self.car.throttle = message.throttle
            elif isinstance(message, SteeringCommand):
                self.car.steering = message.steering


# TODO
# properly close camera?
# check jetracer modules
class ControlModule(BaseModule):
    def init(self):
        self.camera = None

        # import torch here?
        # import torch
        # from torch2trt import TRTModule

        # cuda warmup?
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda()

        self.log("info", "init done")

    def load_model(self, path):
        self.log("info", f"loading model {path}")
        model_trt = TRTModule()
        model_trt.load_state_dict(torch.load(path))
        self.log("info", "Model OK")
        return model_trt

    def preprocess(self, image):
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to tensor
        image = torch.from_numpy(image)
        image = transforms.functional.convert_image_dtype(image, torch.float32)

        # Move channel dimension to the beginning
        image = image.permute(2, 0, 1)
        # test = transforms.functional.to_pil_image(image)
        # test.save("test_pp.jpg")

        # image = transforms.functional.resize(image, (224, 224))
        image = image.cuda().half()

        image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])

        return image[None, ...]

    def handle_image_callback(self, change):
        new_image = change["new"]
        image = self.preprocess(new_image)
        output = self.model(image).squeeze().cpu().numpy()
        traj = output.reshape(-1, 3).astype(np.float32)

        steering = float(traj[self.cfg.lookup, -1]) * self.cfg.steering_gain

        self.publish(SteeringCommand(steering=steering))
        self.publish(SpeedCommand(desired_speed=self.cfg.fixed_speed))

    def reload(self):
        changed_keys = super().reload()

        if "model_path" in changed_keys:
            # pause cam while we load model
            if self.camera is not None:
                self.camera.running = False

            model = self.load_model(self.cfg.model_path)
            self.model = model

            if self.camera is not None:
                self.camera.running = True

        if "camera_framerate" in changed_keys:
            if self.camera is not None:
                self.camera.running = False
                del self.camera

            self.camera = CSICamera(
                width=224,
                height=224,
                capture_width=1280,
                capture_height=720,
                capture_fps=self.cfg.camera_framerate,
            )

            self.camera.observe(self.handle_image_callback, names="value")
            self.camera.running = True
            self.log("info", "SKRRRRT")

    def loop(self):
        while self.running:
            # listen for quit/reload messages
            self.receive()
