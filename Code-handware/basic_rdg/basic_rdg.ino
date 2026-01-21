const int redPin = D0; 
const int greenPin = D1; 
const int bluePin = D2;  

int redColor = 255;
int greenColor = 255;
int blueColor = 255;
void setup() {
  analogWrite(redPin, 0);
  analogWrite(greenPin, 0);
  analogWrite(bluePin, 0);
}

void loop() 
{
    for(int dutyCycle = 0; dutyCycle < 255; dutyCycle++){   
    // changing the LED brightness with PWM
    analogWrite(redPin, dutyCycle);
  analogWrite(greenPin, dutyCycle);
  analogWrite(bluePin, dutyCycle);
    delay(1);
  }

  // decrease the LED brightness
  for(int dutyCycle = 255; dutyCycle > 0; dutyCycle--){
    // changing the LED brightness with PWM
    analogWrite(redPin, dutyCycle);
  analogWrite(greenPin, dutyCycle);
  analogWrite(bluePin, dutyCycle);
    delay(1);
  }
}