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

// Topic nhận lệnh điều khiển (Subscribe)
const char* topic_control_prefix = "esp32/controlled/";
// Topic nhận lệnh yêu cầu gửi lại toàn bộ trạng thái (Subscribe)
const char* topic_get_all = "esp32/get_all";

// Topic gửi trạng thái đi (Publish): esp32/status/0, esp32/status/1...
const char* topic_status_prefix = "esp32/status/";

// ================= WS2812 CONFIG =================
#define LED_PIN     0
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

// ================= HÀM GỬI TRẠNG THÁI 1 LED =================
void publishLedState(int index) {
  if (index < 0 || index >= LED_COUNT) return;

  // Lấy màu hiện tại từ strip (đây là màu ĐÃ tính toán độ sáng)
  uint32_t color = strip.getPixelColor(index);
  uint8_t r = (uint8_t)(color >> 16);
  uint8_t g = (uint8_t)(color >> 8);
  uint8_t b = (uint8_t)color;

  // Tạo chuỗi payload "R,G,B"
  char msg[20];
  sprintf(msg, "%d,%d,%d", r, g, b);

  // Tạo topic status: esp32/status/0, esp32/status/1 ...
  String topicStatus = String(topic_status_prefix) + String(index);

  // Gửi lên MQTT
  client.publish(topicStatus.c_str(), msg);
  // Serial.println("📤 Sent: " + topicStatus + " -> " + String(msg));
}

// ================= HÀM GỬI TRẠNG THÁI TOÀN BỘ LED =================
void publishAllLeds() {
  Serial.println("🔄 Reporting all LEDs status...");
  for (int i = 0; i < LED_COUNT; i++) {
    publishLedState(i);
    delay(20); // Delay nhỏ để tránh nghẽn mạng/tràn buffer MQTT
  }
  Serial.println("✅ Report done.");
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

  // 1. TRƯỜNG HỢP: Yêu cầu lấy toàn bộ trạng thái
  if (topicStr == topic_get_all) {
    publishAllLeds();
    return;
  }

  // 2. TRƯỜNG HỢP: Điều khiển LED lẻ
  // Kiểm tra topic có đúng prefix điều khiển không
  if (!topicStr.startsWith(topic_control_prefix)) return;

  // Lấy index LED
  int ledIndex = topicStr.substring(strlen(topic_control_prefix)).toInt();
  if (ledIndex < 0 || ledIndex >= LED_COUNT) return;

  // Parse payload: R,G,B,Brightness
  int r, g, b, brightness;
  if (sscanf(message.c_str(), "%d,%d,%d,%d", &r, &g, &b, &brightness) == 4) {
    
    // Giới hạn an toàn
    r = constrain(r, 0, 255);
    g = constrain(g, 0, 255);
    b = constrain(b, 0, 255);
    brightness = constrain(brightness, 0, 255);

    // Tính toán màu thực tế
    uint8_t adjustedR = (r * brightness) / 255;
    uint8_t adjustedG = (g * brightness) / 255;
    uint8_t adjustedB = (b * brightness) / 255;
  
    strip.setPixelColor(ledIndex, strip.Color(adjustedR, adjustedG, adjustedB));
    strip.show();

    Serial.printf("✅ LED %d Updated. Reporting status back to MQTT.\n", ledIndex);
    
    // Gửi ngược trạng thái mới lên MQTT để App cập nhật
    publishLedState(ledIndex); 
  }
}

// ================= MQTT RECONNECT =================
void reconnect() {
  while (!client.connected()) {
    Serial.print("Connecting MQTT...");
    String clientId = "ESP32-WS2812-" + String(random(0xffff), HEX);

    if (client.connect(clientId.c_str(), mqtt_username, mqtt_password)) {
      Serial.println("connected");

      // Subscribe topic điều khiển từng LED
      client.subscribe("esp32/controlled/+");
      
      // Subscribe topic yêu cầu lấy trạng thái tất cả
      client.subscribe(topic_get_all);
      
      Serial.println("Subscribed topics ready.");
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
  strip.setBrightness(100); // Lưu ý: Brightness toàn cục giữ 100, ta tính toán tay từng LED

  setup_wifi();
  espClient.setInsecure();

  // Tăng kích thước buffer nếu cần gửi tin nhắn dài (dự phòng)
  client.setBufferSize(512); 
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