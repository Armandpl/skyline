from racecar_inference.proto.protocol_pb2 import (
    FilteredSpeed,
    SpeedCommand,
    SpeedReading,
    SteeringCommand,
    ThrottleCommand,
)

# map message types to int values to send as a single byte on the bus
int_topic_to_message_class = {
    0: None,  # quit command, no need for payload
    1: None,  # reload command, same
    2: SpeedReading,
    3: SpeedCommand,
    4: ThrottleCommand,
    5: SteeringCommand,
    6: FilteredSpeed,
}

# just reverse the dict
message_class_to_int_topic = {
    v: k for k, v in int_topic_to_message_class.items() if k is not None
}

message_class_name_to_int_topic = {
    k.__name__: v for k, v in message_class_to_int_topic.items() if k is not None
}
