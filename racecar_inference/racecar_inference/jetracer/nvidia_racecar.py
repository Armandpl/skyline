import busio
import traitlets
from adafruit_motor.servo import ContinuousServo
from adafruit_pca9685 import PCA9685
from board import SCL, SDA


class NvidiaRacecar(traitlets.HasTraits):

    i2c_address = traitlets.Integer(default_value=0x40)

    # i understand this is what most rc stuff expects
    throttle_min_pulse = 1000
    throttle_max_pulse = 2000

    steering_min_pulse = 1000
    steering_max_pulse = 2000

    # this is bc most rc remotes are not very precise so to avoid having your car move
    # even when the remote is neutral, ECSs implement a dead zone, here 6%
    throttle_dead_zone_pct = (
        6  # configured my xr10 esc so that the dead zone is the smallest
    )

    steering_gain = traitlets.Float(default_value=-1)
    steering_offset = traitlets.Float(default_value=-0.1)
    steering_channel = traitlets.Integer(default_value=0)
    throttle_gain = traitlets.Float(default_value=1.0)
    throttle_channel = traitlets.Integer(default_value=1)

    steering = traitlets.Float(default_value=0)
    throttle = traitlets.Float(default_value=0)

    def __init__(self):
        super().__init__()
        # steering and throttle controls are clipped between -1 and 1
        # servo and esc expect -1, 1 as well so gain can't be >1
        # mainly we use those to e.g don't go past steering travel
        # bc servo can travel farther than the actual steering mechanism
        # TODO we don't really use the throttle gain since we calibrate the ESC?
        # assert steering_gain <= 1.0, "Steering gain must be <= 0"
        # assert throttle_gain <= 1.0, "Throttle gain must be <= 0"
        # would need to check this in on change instead! else it wont check when the value is changed

        self.half_dz = self.throttle_dead_zone_pct / 2 / 100

        i2c = busio.I2C(SCL, SDA)
        pca = PCA9685(i2c, address=self.i2c_address)
        pca.frequency = 50

        self.steering_motor = ContinuousServo(
            pca.channels[self.steering_channel],
            min_pulse=self.steering_min_pulse,
            max_pulse=self.steering_max_pulse,
        )
        self.throttle_motor = ContinuousServo(
            pca.channels[self.throttle_channel],
            min_pulse=self.throttle_min_pulse,
            max_pulse=self.throttle_max_pulse,
        )

    @traitlets.observe("steering")
    def _on_steering(self, change):
        """
        1 right, -1 left
        """
        new_steering = change["new"] * self.steering_gain + self.steering_offset
        if new_steering > 1.0:
            new_steeinrg = 1.0
        elif new_steering < -1.0:
            new_steering = -1
        self.steering_motor.throttle = new_steering

    @traitlets.observe("throttle")
    def _on_throttle(self, change):
        new_throttle = change["new"] * self.throttle_gain

        if new_throttle == 0.0:
            self.throttle_motor.throttle = 0.0
        elif new_throttle > 0.0:  # remap to avoid dead zone
            self.throttle_motor.throttle = self.half_dz + new_throttle * (
                1 - self.half_dz
            )
        elif new_throttle < 0.0:
            self.throttle_motor.throttle = -self.half_dz + new_throttle * (
                1 - self.half_dz
            )

    @traitlets.validate("steering")
    def _clip_steering(self, proposal):
        if proposal["value"] > 1.0:
            return 1.0
        elif proposal["value"] < -1.0:
            return -1.0
        else:
            return proposal["value"]

    @traitlets.validate("throttle")
    def _clip_throttle(self, proposal):
        if proposal["value"] > 1.0:
            return 1.0
        elif proposal["value"] < -1.0:
            return -1.0
        else:
            return proposal["value"]
