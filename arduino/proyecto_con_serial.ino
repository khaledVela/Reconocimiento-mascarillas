int led1 = 2;
int led2 = 10;
int option;
void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pinMode(led1, OUTPUT);
  pinMode(led2, OUTPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  if (Serial.available() > 0) {
    option = Serial.read();
    if (option == 'P') {
      digitalWrite(led1, HIGH);
      digitalWrite(led2, LOW);
      Serial.println("tiene mascara");
    }
    if (option == 'N') {
      digitalWrite(led1, LOW);
      digitalWrite(led2, HIGH);
      Serial.println("no tiene mascara");
    }
    if (option == 'Q') {
      digitalWrite(led1, LOW);
      digitalWrite(led2, LOW);
      Serial.println("Nada");
    }
  }
}
