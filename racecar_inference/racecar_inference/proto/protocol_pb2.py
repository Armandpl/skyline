# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: protocol.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\x0eprotocol.proto\x12\x03\x63\x61r"\x1d\n\x0cSpeedReading\x12\r\n\x05speed\x18\x01 \x01(\x02"%\n\x0cSpeedCommand\x12\x15\n\rdesired_speed\x18\x01 \x01(\x02"#\n\x0fThrottleCommand\x12\x10\n\x08throttle\x18\x01 \x01(\x02"#\n\x0fSteeringCommand\x12\x10\n\x08steering\x18\x01 \x01(\x02\x62\x06proto3'
)


_SPEEDREADING = DESCRIPTOR.message_types_by_name["SpeedReading"]
_SPEEDCOMMAND = DESCRIPTOR.message_types_by_name["SpeedCommand"]
_THROTTLECOMMAND = DESCRIPTOR.message_types_by_name["ThrottleCommand"]
_STEERINGCOMMAND = DESCRIPTOR.message_types_by_name["SteeringCommand"]
SpeedReading = _reflection.GeneratedProtocolMessageType(
    "SpeedReading",
    (_message.Message,),
    {
        "DESCRIPTOR": _SPEEDREADING,
        "__module__": "protocol_pb2"
        # @@protoc_insertion_point(class_scope:car.SpeedReading)
    },
)
_sym_db.RegisterMessage(SpeedReading)

SpeedCommand = _reflection.GeneratedProtocolMessageType(
    "SpeedCommand",
    (_message.Message,),
    {
        "DESCRIPTOR": _SPEEDCOMMAND,
        "__module__": "protocol_pb2"
        # @@protoc_insertion_point(class_scope:car.SpeedCommand)
    },
)
_sym_db.RegisterMessage(SpeedCommand)

ThrottleCommand = _reflection.GeneratedProtocolMessageType(
    "ThrottleCommand",
    (_message.Message,),
    {
        "DESCRIPTOR": _THROTTLECOMMAND,
        "__module__": "protocol_pb2"
        # @@protoc_insertion_point(class_scope:car.ThrottleCommand)
    },
)
_sym_db.RegisterMessage(ThrottleCommand)

SteeringCommand = _reflection.GeneratedProtocolMessageType(
    "SteeringCommand",
    (_message.Message,),
    {
        "DESCRIPTOR": _STEERINGCOMMAND,
        "__module__": "protocol_pb2"
        # @@protoc_insertion_point(class_scope:car.SteeringCommand)
    },
)
_sym_db.RegisterMessage(SteeringCommand)

if _descriptor._USE_C_DESCRIPTORS == False:

    DESCRIPTOR._options = None
    _SPEEDREADING._serialized_start = 23
    _SPEEDREADING._serialized_end = 52
    _SPEEDCOMMAND._serialized_start = 54
    _SPEEDCOMMAND._serialized_end = 91
    _THROTTLECOMMAND._serialized_start = 93
    _THROTTLECOMMAND._serialized_end = 128
    _STEERINGCOMMAND._serialized_start = 130
    _STEERINGCOMMAND._serialized_end = 165
# @@protoc_insertion_point(module_scope)
