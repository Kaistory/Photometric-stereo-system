#include <Adafruit_NeoPixel.h>

#define LED_PIN    0
#define LED_COUNT 40

// --- CHỈNH MÀU TẠI ĐÂY ---
byte R = 255; 
byte G = 100; 
byte B = 0;   
// -----------------------

Adafruit_NeoPixel strip(LED_COUNT, LED_PIN, NEO_GRB + NEO_KHZ800);

void setup() {
  Serial.begin(115200);
  strip.begin();
  strip.setBrightness(100); // Thêm độ sáng (0-255) để bảo vệ mắt và nguồn
  strip.clear();
  strip.show();

  Serial.println("Dang test LED...");

  for (int i = 0; i < 3; i++) {
    // Bật tất cả LED theo màu đã chọn
    for (int j = 0; j < LED_COUNT; j++) {
      strip.setPixelColor(j, strip.Color(R, G, B)); 
    }
    strip.show();
    delay(500);

    // Tắt LED
    strip.clear();
    strip.show();
    delay(500);
  }

  Serial.println("Test xong!");
}

void loop() {
  // Nếu muốn LED sáng đứng yên sau khi test, bỏ comment dòng dưới:
  // /*
  for (int j = 0; j < LED_COUNT; j++) {
      strip.setPixelColor(j, strip.Color(R, G, B));
  }
  strip.show();
  
}