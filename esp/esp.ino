#include <ESP8266WiFi.h>

// Define motor and LED control pins
const int motorPin1 = 5;  // Motor IN1 connected to GPIO 5
const int motorPin2 = 4;  // Motor IN2 connected to GPIO 4
const int enablePin = 14; // Motor EN connected to GPIO 14
const int ledPin = 12;    // LED connected to GPIO 12

// Wi-Fi credentials
const char *ssid = "Your_SSID";
const char *password = "Your_PASSWORD";

// Web server on port 80
WiFiServer server(80);

// Global variable to keep track of the current motor speed
float current_motor_speed = 0.0;

void setup() {
  // Initialize serial communication
  Serial.begin(115200);

  // Set motor and LED control pins as outputs
  pinMode(motorPin1, OUTPUT);
  pinMode(motorPin2, OUTPUT);
  pinMode(enablePin, OUTPUT);
  pinMode(ledPin, OUTPUT);

  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED)
  {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi connected.");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());

  // Start the server
  server.begin();
}

void loop()
{
  // Check if a client has connected
  WiFiClient client = server.available();
  if (client)
  {
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

    // Handle motor control
    if (request.indexOf("MOTOR=") != -1) {
      int index = request.indexOf("MOTOR=") + 6;
      String speedString = request.substring(index);
      if (speedString == "FASTER") {
        current_motor_speed = min(current_motor_speed + 0.2, 1.0);
      } else if (speedString == "SLOWER") {
        current_motor_speed = max(current_motor_speed - 0.2, 0.0);
      } else {
        current_motor_speed = speedString.toFloat();
      }

      // Map the speed value (0.0 to 1.0) to PWM range (0 to 255)
      int pwmValue = int(current_motor_speed * 255);

      // Set motor direction and speed
      digitalWrite(motorPin1, HIGH); // Set direction
      digitalWrite(motorPin2, LOW);  // Set direction
      analogWrite(enablePin, pwmValue);

      Serial.print("Motor speed set to: ");
      Serial.println(current_motor_speed * 100); // Show as percentage
    }

    // Send an HTTP response to the client
    client.print("HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n");
    client.print("<!DOCTYPE HTML>\r\n<html>\r\n");
    client.print("<h1>ESP8266 Motor and LED Control</h1>");
    client.print("<p>Control the motor speed by appending '?MOTOR=0.5' to the URL</p>");
    client.print("<p>Control the LED by appending '/LED=1' or '/LED=0' to the URL</p>");
    client.print("</html>\n");
    delay(1);
    Serial.println("Client disconnected");
  }
}