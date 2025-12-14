#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <PubSubClient.h>
#include <Adafruit_NeoPixel.h>

// ================= WIFI & MQTT =================
const char* ssid = "TP-Link_847E";
const char* password = "39926227";

const char* mqtt_server = "76bf78e5731240cbb090e3284089c085.s1.eu.hivemq.cloud";
const int mqtt_port = 8883;
const char* mqtt_username = "hivemq.webclient.1765688353696";
const char* mqtt_password = "r8>!lHMZ37RYzy?5<Sxv";

// Topic gốc
const char* topic_control_prefix = "esp32/controlled/";

// ================= WS2812 CONFIG =================
#define LED_PIN     5
#define LED_COUNT   24

Adafruit_NeoPixel strip(LED_COUNT, LED_PIN, NEO_GRB + NEO_KHZ800);

// ================= MQTT =================
WiFiClientSecure espClient;
PubSubClient client(espClient);

// ================= WIFI =================
void setup_wifi() {
  Serial.print("Connecting to WiFi...");
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi connected");
  Serial.println(WiFi.localIP());
}

// ================= MQTT CALLBACK =================
void callback(char* topic, byte* payload, unsigned int length) {
  String topicStr = topic;
  String message = "";

  for (int i = 0; i < length; i++) {
    message += (char)payload[i];
  }

  Serial.println("📩 MQTT:");
  Serial.println("Topic: " + topicStr);
  Serial.println("Payload: " + message);

  // Kiểm tra topic có đúng prefix không
  if (!topicStr.startsWith(topic_control_prefix)) return;

  // Lấy index LED
  int ledIndex = topicStr.substring(strlen(topic_control_prefix)).toInt();
  if (ledIndex < 0 || ledIndex >= LED_COUNT) return;

  // Parse payload: R,G,B,Brightness
  int r, g, b, brightness;
  sscanf(message.c_str(), "%d,%d,%d,%d", &r, &g, &b, &brightness);

  // Giới hạn an toàn
  r = constrain(r, 0, 255);
  g = constrain(g, 0, 255);
  b = constrain(b, 0, 255);
  brightness = constrain(brightness, 0, 255);

  strip.setBrightness(brightness);
  strip.setPixelColor(ledIndex, strip.Color(r, g, b));
  strip.show();

  Serial.printf("✅ LED %d -> R:%d G:%d B:%d Bright:%d\n",
                ledIndex, r, g, b, brightness);
}

// ================= MQTT RECONNECT =================
void reconnect() {
  while (!client.connected()) {
    Serial.print("Connecting MQTT...");
    String clientId = "ESP32-WS2812-";
    clientId += String(random(0xffff), HEX);

    if (client.connect(clientId.c_str(), mqtt_username, mqtt_password)) {
      Serial.println("connected");

      // Subscribe toàn bộ LED
      client.subscribe("esp32/controlled/+");
      Serial.println("Subscribed: esp32/controlled/+");
    } else {
      Serial.print("failed rc=");
      Serial.println(client.state());
      delay(2000);
    }
  }
}

// ================= SETUP =================
void setup() {
  Serial.begin(115200);

  strip.begin();
  strip.clear();
  strip.show();
  strip.setBrightness(100);

  setup_wifi();
  espClient.setInsecure();

  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);

  Serial.println("🚀 ESP32 WS2812 READY");
}

// ================= LOOP =================
void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();
}
