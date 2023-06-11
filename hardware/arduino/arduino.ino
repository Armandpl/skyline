const byte A = 8;
const byte B = 9;
const byte C = 10;

const float diam_mm = 66;
const float wheel_perimeter_m = 2 * 3.1416 * diam_mm/1000/2;
const float tick_per_motor_rotation = 6;
const float motor_rotations_per_wheel_turn = 5.65;
const float tick_per_wheel_rotation = tick_per_motor_rotation*motor_rotations_per_wheel_turn;
const float freq = 50;
const float t_s = 1/freq;
const float t_ms = 1000/freq;

volatile unsigned long last_time = 0;
volatile float speed = 0;

// .5s, TODO compute it from the min speed of the car
// if more than theshold between two interrupts, the car is probably stopped
// don't compute the speed
volatile float threshold = 5e5;

// keep last interrupt time
// if last interrupt is very old car is probably stopped, set speed to zero
// TODO again compute this with the min speed of the car, what's the max dt
// maybe use twice that as the timeout just in case
volatile unsigned long last_interrupt_time = 0;
const unsigned long timeout_us = 5e4; // 0.05s

void setup() {
  attachInterrupt(digitalPinToInterrupt(A), incrementCounter, CHANGE);
  attachInterrupt(digitalPinToInterrupt(B), incrementCounter, CHANGE);
  attachInterrupt(digitalPinToInterrupt(C), incrementCounter, CHANGE);

  Serial.begin(9600);
}

void loop() {
  delay(t_ms);

  unsigned long current_time = micros();

  // Check if the timeout has been exceeded
  // if so set the speed to zero, means the car is stopped
  if ((current_time - last_interrupt_time) > timeout_us) {
    speed = 0;
  }

  Serial.println(speed);
}

void incrementCounter() {
  unsigned long current_time = micros();
  unsigned long dt_micros = current_time - last_time;

  // makes sure we only compute speed when we have at least to consecutive interrupts
  // else the first value we compute is very big
  // TODO why is it big actually?
  if (dt_micros < threshold) {
    float dt_s = dt_micros / 1e6;
    float wheel_rotations = 1 / tick_per_wheel_rotation;
    float metters = wheel_rotations * wheel_perimeter_m;
    speed = metters / dt_s;
  }

  last_time = current_time;
  last_interrupt_time = current_time; // Save the time of the last interrupt
}
