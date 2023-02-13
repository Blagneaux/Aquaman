#include <Servo.h>
#include <Stepper.h>

//Stepper motor
#define stepsPerRevolution 200 //1.8° = 200 steps/turn

// Give the motor control pins names:
#define pwmA 3
#define pwmB 11
#define brakeA 9
#define brakeB 8
#define dirA 12
#define dirB 13
Stepper myStepper = Stepper(stepsPerRevolution, dirA, dirB);

//Switches pins
#define switchPin01 33 // back of the tank
#define switchPin02 32 // carrier-gantry contact
#define switchPin03 35 // front of the tank
#define switchNotTouch LOW //input level when switch is not activated

//Servomotor
#define servoPin 5 // Flapping servomotor
Servo flapServo;
boolean isFlapping = true;
boolean isHome = true;
boolean isEnd = false;
float flapPos = 0.0;
boolean isStepRun = false;
int travelLength = 2400;

// Parameters to command the two motors almost independently
unsigned long previousMotorTime = millis();
long MotorInterval = 10;

void setup() {
  // Set the PWM and brake pins so that the direction pins can be used to control the motor:
  pinMode(pwmA, OUTPUT);
  pinMode(pwmB, OUTPUT);
  pinMode(brakeA, OUTPUT);
  pinMode(brakeB, OUTPUT);
  digitalWrite(pwmA, HIGH);
  digitalWrite(pwmB, HIGH);
  digitalWrite(brakeA, LOW);
  digitalWrite(brakeB, LOW);

  // Servo initialization
  flapServo.attach(servoPin);

  pinMode(switchPin01, INPUT_PULLUP);
  pinMode(switchPin02, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(switchPin02), touch02, LOW);
  pinMode(switchPin03, INPUT_PULLUP); 

  delay(200);// wait 200ms
}

void loop() {
  // Set the motor speed (RPMs):
  myStepper.setSpeed(150);

  if ((isEnd == false or isHome == true) and digitalRead(switchPin03) != switchNotTouch){
    // we are moving
    isStepRun = true;
    float t = 0;

    while (digitalRead(switchPin01) == switchNotTouch){ // while we don't touch the back
      if(digitalRead(switchPin01) != switchNotTouch) {
        break;
      }

      unsigned long currentMotorTime = millis();

      myStepper.step(-2);
    
      if (currentMotorTime - previousMotorTime > MotorInterval){
        int angle = 90 + 22.5 * sin(t * 2 * PI / 1.225);           // calculate the angle based on the sinusoidal signal, offset by 90 degrees
        angle = constrain(angle, 68.5, 122.5);                       // ensure the angle stays within the range of 60 to 120 degrees
        flapServo.write(angle);                                  // set the servo angle
        previousMotorTime = currentMotorTime;
        t += 0.01215;
      }
    }    

    delay(500);

    // Finally, move 10mm forward (10mm = 55.6 steps)
    myStepper.setSpeed(100);
    myStepper.step(56);
    

    isStepRun = false;
    isEnd = true;
    isHome = false;

    delay(2000);
  }
   
  if ((isEnd == true or isHome == false) and digitalRead(switchPin01) != switchNotTouch){
    flapServo.write(90);  
    // Set the motor speed (RPMs):
    myStepper.setSpeed(150);

    // we are moving
    isStepRun = true;
    
    while (digitalRead(switchPin03) == switchNotTouch){ // while we don't touch the front
      myStepper.step(2);
    }

    delay(500);

    // Finally, move 10mm backward (10mm = 55.6 steps)
    myStepper.setSpeed(100);
    myStepper.step(-56);

    isStepRun = false;
    isEnd = false;
    isHome = true;

    delay(2000);
  }
}

void touch02() {
  isEnd = true;
  isHome = false;
  delay(20);
}
