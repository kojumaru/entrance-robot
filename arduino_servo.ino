#include <Servo.h>

Servo servo3;
Servo servo5;
Servo servo10;
Servo servo11;

const int PIN_SERVO_3  = 3;
const int PIN_SERVO_5  = 5;
const int PIN_SERVO_10 = 10;
const int PIN_SERVO_11 = 11;

int mode = 0;

unsigned long lastMode0Time = 0;
unsigned long lastMode1Time = 0;

const unsigned long MODE0_INTERVAL = 1000;
const unsigned long MODE1_INTERVAL = 2000;

int angle3 = 90;
int angle5 = 90;
int angle10 = 90;
int angle11 = 90;

void moveServos(Servo& s1, int pin1, int a1, Servo& s2, int pin2, int a2) {
  s1.attach(pin1);
  s2.attach(pin2);
  delay(100);
  s1.write(a1);
  s2.write(a2);
  delay(700);
  s1.detach();
  s2.detach();
  delay(100);
}

void setup() {
  Serial.begin(115200);
  randomSeed(analogRead(A0));

  moveServos(servo3, PIN_SERVO_3, angle3, servo5, PIN_SERVO_5, angle5);
  moveServos(servo10, PIN_SERVO_10, angle10, servo11, PIN_SERVO_11, angle11);

  printStatus();
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    int newMode = input.toInt();
    if (newMode == 0 || newMode == 1) {
      mode = newMode;
      Serial.print("MODE CHANGED: ");
      Serial.println(mode);
      printStatus();
    }
  }

  unsigned long currentTime = millis();

  if (mode == 0) {
    if (currentTime - lastMode0Time >= MODE0_INTERVAL) {
      lastMode0Time = currentTime;
      angle3 = random(0, 121);
      angle5 = random(0, 121);
      moveServos(servo3, PIN_SERVO_3, angle3, servo5, PIN_SERVO_5, angle5);
      printStatus();
    }
  } else if (mode == 1) {
    if (currentTime - lastMode1Time >= MODE1_INTERVAL) {
      lastMode1Time = currentTime;
      angle10 = random(0, 101);
      angle11 = random(0, 101);
      moveServos(servo10, PIN_SERVO_10, angle10, servo11, PIN_SERVO_11, angle11);
      printStatus();
    }
  }
}

void printStatus() {
  Serial.print("MODE: ");
  Serial.print(mode);
  Serial.print(" | S3: ");
  Serial.print(angle3);
  Serial.print(" | S5: ");
  Serial.print(angle5);
  Serial.print(" | S10: ");
  Serial.print(angle10);
  Serial.print(" | S11: ");
  Serial.println(angle11);
}
