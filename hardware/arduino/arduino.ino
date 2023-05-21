const byte A = 8;
const byte B = 9;
const byte C = 10;

const float diam = 66;
const float wheel_perimeter = 2 * 3.1416 * 66/1000/2;
const float tick_per_motor_rotation = 4;
const float motor_rotations_per_wheel_turn = 5.65;
const float tick_per_wheel_rotation = tick_per_motor_rotation*motor_rotations_per_wheel_turn;
const float freq = 40;
const float t = 1/freq;

volatile int counter = 0;

void setup() {
  attachInterrupt(digitalPinToInterrupt(B), incrementCounter, CHANGE);
  attachInterrupt(digitalPinToInterrupt(B), incrementCounter, CHANGE);
  attachInterrupt(digitalPinToInterrupt(C), incrementCounter, CHANGE);

  Serial.begin(9600);
}

void loop() {
  delay(t*1000);

  float wheel_rotations = counter / tick_per_wheel_rotation;
  float metters = wheel_rotations * wheel_perimeter;
  float metters_per_sec = metters / t;
  Serial.println(metters_per_sec);

  counter = 0;
}

void incrementCounter() {
  counter += 1;
}
