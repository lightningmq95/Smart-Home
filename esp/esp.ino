#include <ESP8266WiFi.h>

// Define motor and LED control pins
const int motorPin1 = 12;  // Motor IN1 connected to GPIO 5
const int motorPin2 = 13;  // Motor IN2 connected to GPIO 4
const int enablePin = 14;  // Motor EN connected to GPIO 14
const int ledPin = 2;      // LED connected to GPIO 2 (not 12)

// Wi-Fi credentials
const char *ssid = "Akash";
const char *password = "Akash@3010";
const uint16_t port = 8002;
const char *host = "192.168.1.6";

// Web server on port 80
WiFiServer server(port);

// Current motor speed state
float currentSpeed = 0.0;

void setup() {
  // Initialize serial communication
  Serial.begin(9600);

  // Set motor and LED control pins as outputs
  pinMode(motorPin1, OUTPUT);
  pinMode(motorPin2, OUTPUT);
  pinMode(enablePin, OUTPUT);
  pinMode(ledPin, OUTPUT);

  Serial.print("Connecting to: ");
  Serial.println(ssid);

  WiFi.begin(ssid, password);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.println("...");
    // attempts++;
    // if (attempts > 10) {
    //   ESP.restart();
    // }
  }
  Serial.print("WiFi connected with IP: ");
  Serial.println(WiFi.localIP());

  server.begin();
}

void loop() {
  // Check if a client has connected
  WiFiClient client = server.available();
  if (client) {
    Serial.println("New Client connected");
    String request = client.readStringUntil('\r');
    Serial.println("Request: " + request);
    client.flush();

    // Handle LED control
    if (request.indexOf("/LED=1") != -1) {
      digitalWrite(ledPin, HIGH);
      Serial.println("LED is ON");
    } else if (request.indexOf("/LED=0") != -1) {
      digitalWrite(ledPin, LOW);
      Serial.println("LED is OFF");
    }

    // Parse the motor speed value from the request
    if (request.indexOf("MOTOR=") != -1) {
      int index = request.indexOf("MOTOR=") + 6;
      String speedString = request.substring(index);
      int endIndex = speedString.indexOf(' ');
      if (endIndex != -1) {
        speedString = speedString.substring(0, endIndex);
      }
      speedString.trim();  // Remove any leading/trailing whitespace
      Serial.println("Parsed speedString: '" + speedString + "'");

      // Adjust motor speed based on the received value
      if (speedString == "1") {
        currentSpeed = 1.0;  // Max speed
        Serial.println("Setting motor speed to max (1.0)");
      } else if (speedString == "0") {
        currentSpeed = 0.0;  // Motor off
        Serial.println("Turning motor off (0.0)");
      } else if (speedString == "FASTER") {
        currentSpeed += 0.2;  // Increase speed
        if (currentSpeed > 1.0)
          currentSpeed = 1.0;  // Clamp to max speed
        Serial.println("Increasing motor speed");
      } else if (speedString == "SLOWER") {
        currentSpeed -= 0.2;  // Decrease speed
        if (currentSpeed < 0.0)
          currentSpeed = 0.0;  // Clamp to min speed
        Serial.println("Decreasing motor speed");
      } else {
        Serial.println("Invalid motor speed command");
      }

      // Map the speed value (0.0 to 1.0) to PWM range (0 to 255)
      int pwmValue = int(currentSpeed * 255);

      // Set motor direction and speed
      digitalWrite(motorPin1, HIGH);  // Set direction
      digitalWrite(motorPin2, LOW);   // Set direction
      analogWrite(enablePin, pwmValue);

      Serial.print("Motor speed set to: ");
      Serial.println(currentSpeed * 100);  // Show as percentage
    }

    // Send an HTTP response to the client
    client.print("HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n");
    client.print("<!DOCTYPE HTML>\r\n<html>\r\n");
    client.print("<h1>ESP8266 Motor and LED Control</h1>");
    client.print("<p>Control the motor speed by appending '?MOTOR=1' for max speed, '?MOTOR=0' for off, '?MOTOR=FASTER' to increase speed, or '?MOTOR=SLOWER' to decrease speed</p>");
    client.print("<p>Control the LED by appending '/LED=1' or '/LED=0' to the URL</p>");
    client.print("</html>\n");
    delay(1);
    Serial.println("Client disconnected");
  }
}
