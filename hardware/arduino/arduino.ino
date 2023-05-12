const byte A = D8;
const byte B = D9;
const byte C = D10;

const float diam = 66;
const float wheel_perimeter = 2 * 3.1416 * 66/1000/2;
const float tick_per_motor_rotation = 6;
const float motor_rotations_per_wheel_turn = 5.65;
const float tick_per_wheel_rotation = ticke_per_motor_rotation*motor_rotations_per_wheel_turn;
const float freq = 40;
const float t = 1/freq;

volatile int counter = 0;

void setup() {
  attachInterrupt(digitalPinToInterrupt(A), incrementCounter, RISING);
  attachInterrupt(digitalPinToInterrupt(B), incrementCounter, RISING);
  attachInterrupt(digitalPinToInterrupt(C), incrementCounter, RISING);

  Serial.begin(9600);
}

void loop() {
  delay(t*1000);

  float wheel_rotations = counter / tick_per_wheel_rotation;
  float metters = wheel_rotations * wheel_perimeter;
  float metters_per_sec = metters / t;
  // Serial.println(metters_per_sec);
  Serial.println(counter);

  //counter = 0;
}

void incrementCounter() {
  counter += 1;
}