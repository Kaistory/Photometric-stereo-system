#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <Adafruit_NeoPixel.h>

// ================= WS2812 CONFIG =================
#define LED_PIN     0
#define LED_COUNT   40

Adafruit_NeoPixel strip(LED_COUNT, LED_PIN, NEO_GRB + NEO_KHZ800);

// ================= BLE CONFIG =================
BLEServer *pServer = NULL;
BLECharacteristic *pTxCharacteristic;
bool deviceConnected = false;
bool oldDeviceConnected = false;
uint8_t txValue = 0;

#define SERVICE_UUID           "6E400001-B5A3-F393-E0A9-E50E24DCCA9E" 
#define CHARACTERISTIC_UUID_RX "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
#define CHARACTERISTIC_UUID_TX "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

// ================= OLED CONFIG =================

#define SCREEN_WIDTH 128 
#define SCREEN_HEIGHT 64 
#define I2C_SDA 8
#define I2C_SCL 9
#define OLED_RESET -1 
#define SCREEN_ADDRESS 0x3C 

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// ================= WS2812 SETUP =================

void setupLed(){
  strip.begin();
  strip.setBrightness(100); // Thêm độ sáng (0-255) để bảo vệ mắt và nguồn
  strip.clear();
  strip.show();
}

// ================= LED DISPLAY =================

uint32_t Wheel(byte WheelPos) {
  WheelPos = 255 - WheelPos;
  if (WheelPos < 85) {
    return strip.Color(255 - WheelPos * 3, 0, WheelPos * 3);
  }
  if (WheelPos < 170) {
    WheelPos -= 85;
    return strip.Color(0, WheelPos * 3, 255 - WheelPos * 3);
  }
  WheelPos -= 170;
  return strip.Color(WheelPos * 3, 255 - WheelPos * 3, 0);
}

/**
 * @brief Hiển thị hiệu ứng cầu vồng chạy qua một lượt rồi tắt
 * @param wait Độ trễ giữa mỗi khung hình (ms) - càng nhỏ chạy càng nhanh
 */
void showRainbowAndOff(uint8_t wait) {
  // Chạy hiệu ứng cầu vồng (256 sắc độ)
  for (long firstPixelHue = 0; firstPixelHue < 256; firstPixelHue++) {
    for (int i = 0; i < strip.numPixels(); i++) {
      // Tính toán màu cho từng LED dựa trên vị trí của nó
      strip.setPixelColor(i, Wheel((i * 256 / strip.numPixels() + firstPixelHue) & 255));
    }
    strip.show();
    delay(wait);
  }

  // Tắt dần hoặc tắt ngay lập tức
  Serial.println("Rainbow finished - Turning off.");
  strip.clear();
  strip.show();
}

void showPoliceStrobeAndOff(int count, uint8_t wait) {
  for (int a = 0; a < count; a++) {
    // Nửa đầu Đỏ, nửa sau Xanh
    for (int i = 0; i < LED_COUNT; i++) {
      if (i < LED_COUNT / 2) strip.setPixelColor(i, 255, 0, 0);
      else strip.setPixelColor(i, 0, 0, 255);
    }
    strip.show();
    delay(wait);

    // Đảo ngược lại: Nửa đầu Xanh, nửa sau Đỏ
    for (int i = 0; i < LED_COUNT; i++) {
      if (i < LED_COUNT / 2) strip.setPixelColor(i, 0, 0, 255);
      else strip.setPixelColor(i, 255, 0, 0);
    }
    strip.show();
    delay(wait);
  }
  
  strip.clear();
  strip.show();
  Serial.println("Police Strobe finished.");
}

void setPixelColorWithBrightness(int index, uint8_t r, uint8_t g, uint8_t b, uint8_t brightness) {
    // 1. Kiểm tra giới hạn index để tránh crash chương trình (quan trọng trong IoT)
    if (index < 0 || index >= LED_COUNT) return;

    // 2. Tính toán màu sắc dựa trên độ sáng (Scaling)
    // Dùng uint16_t để tránh tràn số khi nhân trước khi chia
    uint8_t adjR = (uint16_t(r) * brightness) / 255;
    uint8_t adjG = (uint16_t(g) * brightness) / 255;
    uint8_t adjB = (uint16_t(b) * brightness) / 255;

    // 3. Đẩy dữ liệu vào buffer của strip
    strip.setPixelColor(index, strip.Color(adjR, adjG, adjB));
    
    // 4. Hiển thị
    strip.show();

    // Debug nhanh
    // Serial.printf("LED %d updated\n", index);
}

// ================= BLE SOLVE =================

class MyServerCallbacks : public BLEServerCallbacks {
    void onConnect(BLEServer *pServer) {
      deviceConnected = true;
      Serial.println("Đã kết nối thiết bị");
      showRainbowAndOff(10);
    };

    void onDisconnect(BLEServer *pServer) {
      deviceConnected = false;
      Serial.println("Đã ngắt kết nối");
      showPoliceStrobeAndOff(1,1);
    }
};

class MyCallbacks : public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic* pCharacteristic) {
        String rxRaw = pCharacteristic->getValue();

        if (rxRaw.length() > 0) {
            int idx, r, g, b, bright;
            // Bóc tách 5 tham số từ chuỗi App gửi xuống
            int count = sscanf(rxRaw.c_str(), "%d,%d,%d,%d,%d", &idx, &r, &g, &b, &bright);

            uint8_t response[2]; // Gói tin phản hồi: [Mã trạng thái, Số thứ tự LED]

            if (count == 5) {
                if (idx >= 0 && idx < LED_COUNT) {
                    // --- THÀNH CÔNG ---
                    setPixelColorWithBrightness(idx, (uint8_t)r, (uint8_t)g, (uint8_t)b, (uint8_t)bright);
                    
                    response[0] = 0x00;           // 0x00 nghĩa là OK
                    response[1] = (uint8_t)idx;   // Trả lại đúng số thứ tự LED vừa bật
                } else {
                    // --- LỖI INDEX (Vượt quá số LED) ---
                    response[0] = 0x01;           // 0x01 nghĩa là lỗi vị trí
                    response[1] = (uint8_t)idx;   // Trả lại cái index sai để App biết LED nào lỗi
                }
            } else {
                // --- LỖI ĐỊNH DẠNG (Gửi thiếu số) ---
                response[0] = 0x02;               // 0x02 nghĩa là lỗi dữ liệu
                response[1] = 255;                // Giá trị tượng trưng (n/a)
            }

            // Gửi mảng 2 byte này về điện thoại qua cổng TX
            pTxCharacteristic->setValue(response, 2); 
            pTxCharacteristic->notify(); 

            Serial.printf("Xác thực: Status %d | LED %d\n", response[0], response[1]);
        }
    }
};

// ================= BLE SETUP =================

void setupBLE(){
  // Khởi tạo BLE
  BLEDevice::init("ESP32_LED_Control"); // Tên thiết bị hiển thị trên điện thoại

  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);

  pTxCharacteristic = pService->createCharacteristic(
                        CHARACTERISTIC_UUID_TX,
                        BLECharacteristic::PROPERTY_NOTIFY
                      );
  pTxCharacteristic->addDescriptor(new BLE2902());

  BLECharacteristic *pRxCharacteristic = pService->createCharacteristic(
                                         CHARACTERISTIC_UUID_RX,
                                         BLECharacteristic::PROPERTY_WRITE 
                                       );

  pRxCharacteristic->setCallbacks(new MyCallbacks());

  pService->start();
  pServer->getAdvertising()->start();
  Serial.println("Đang chờ điện thoại kết nối...");
}

// ================= OLED SETUP =================

void setupOLED(){
  Wire.begin(I2C_SDA, I2C_SCL);

  if(!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
    for(;;); // Dừng nếu không tìm thấy màn hình
  }

  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  // Hiển thị dòng 1: "HOW ARE"
  display.setTextSize(2);      // Chữ to mức 2
  display.setCursor(20, 10);   // Tọa độ X=20, Y=10
  display.println(F("HOW ARE"));

  // Hiển thị dòng 2: "HUSTER?"
  display.setTextSize(2);      
  display.setCursor(20, 35);   // Xuống dòng (Y=35)
  display.println(F("HUSTER?"));

  display.display();
}

void setup() {
  Serial.begin(115200);
  
  setupLed();

  setupBLE();

  setupOLED();
}

void loop() {
  // Phản hồi trạng thái lên điện thoại mỗi giây (tùy chọn)
    if (deviceConnected) {
        // Bạn có thể gửi dữ liệu ngược lại điện thoại ở đây nếu muốn
    }

    // Xử lý khi ngắt kết nối (tự động phát quảng cáo lại)
    if (!deviceConnected && oldDeviceConnected) {
        delay(500); 
        pServer->startAdvertising(); 
        Serial.println("Đang phát quảng cáo lại...");
        oldDeviceConnected = false;
    }
    if (deviceConnected && !oldDeviceConnected) {
        oldDeviceConnected = true;
    }
}
