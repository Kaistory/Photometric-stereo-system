#include <Arduino.h>
#include <Adafruit_NeoPixel.h>

// Cấu hình LED WS2812
#define LED_PIN 5     // Chân GPIO kết nối với LED (có thể thay đổi)
#define LED_COUNT 24  // Số lượng LED trong dây
#define BRIGHTNESS 50 // Độ sáng (0-255)

// Khởi tạo đối tượng NeoPixel
Adafruit_NeoPixel strip(LED_COUNT, LED_PIN, NEO_GRB + NEO_KHZ800);

void setup()
{
  Serial.begin(115200);
  Serial.println("Khởi động WS2812 LED Strip...");

  // Khởi tạo LED strip
  strip.begin();
  strip.setBrightness(BRIGHTNESS);
  strip.show(); // Tắt tất cả LED ban đầu

  Serial.println("LED Strip đã sẵn sàng!");
}

// ------------------------------------------
// HÀM ĐIỀU KHIỂN TỪNG LED RIÊNG LẺ
// ------------------------------------------
/**
 * Đặt màu cho một LED cụ thể và hiển thị nó.
 * @param ledIndex Chỉ số của LED (từ 0 đến LED_COUNT - 1).
 * @param color Màu 32-bit (ví dụ: strip.Color(R, G, B)).
 */
void setSingleLed(uint16_t ledIndex, uint32_t color)
{
  if (ledIndex < strip.numPixels())
  {
    strip.setPixelColor(ledIndex, color);
    strip.show(); // Hiển thị thay đổi ngay lập tức
  }
}

// ------------------------------------------
// CÁC HIỆU ỨNG HIỆN CÓ
// ------------------------------------------

// Hiệu ứng tô màu từng LED
void colorWipe(uint32_t color, int wait)
{
  for (int i = 0; i < strip.numPixels(); i++)
  {
    strip.setPixelColor(i, color);
    strip.show();
    delay(wait);
  }
}

// Hiệu ứng rạp chiếu phim
void theaterChase(uint32_t color, int wait)
{
  for (int j = 0; j < 10; j++)
  {
    for (int q = 0; q < 3; q++)
    {
      for (int i = 0; i < strip.numPixels(); i = i + 3)
      {
        strip.setPixelColor(i + q, color);
      }
      strip.show();
      delay(wait);

      for (int i = 0; i < strip.numPixels(); i = i + 3)
      {
        strip.setPixelColor(i + q, 0);
      }
    }
  }
}

// Hàm tạo màu từ bánh xe màu
uint32_t wheel(byte WheelPos)
{
  WheelPos = 255 - WheelPos;
  if (WheelPos < 85)
  {
    return strip.Color(255 - WheelPos * 3, 0, WheelPos * 3);
  }
  if (WheelPos < 170)
  {
    WheelPos -= 85;
    return strip.Color(0, WheelPos * 3, 255 - WheelPos * 3);
  }
  WheelPos -= 170;
  return strip.Color(WheelPos * 3, 255 - WheelPos * 3, 0);
}

void rainbowCycle(int wait)
{
  for (int j = 0; j < 256; j++)
  {
    for (int i = 0; i < strip.numPixels(); i++)
    {
      strip.setPixelColor(i, wheel((i * 256 / strip.numPixels() + j) & 255));
    }
    strip.show();
    delay(wait);
  }
}
uint32_t dimColor(uint32_t color, float factor) 
{
    // Đảm bảo hệ số nằm trong khoảng [0.0, 1.0]
    if (factor > 1.0) factor = 1.0;
    if (factor < 0.0) factor = 0.0;

    // Tách các thành phần màu R, G, B từ màu 32-bit
    uint8_t r = (uint8_t)(color >> 16);
    uint8_t g = (uint8_t)(color >>  8);
    uint8_t b = (uint8_t)(color >>  0);

    // Điều chỉnh (nhân) từng thành phần với hệ số
    r = (uint8_t)(r * factor);
    g = (uint8_t)(g * factor);
    b = (uint8_t)(b * factor);

    // Trả về màu 32-bit mới đã làm mờ
    return strip.Color(r, g, b);
}
// ------------------------------------------
// HÀM CHẠY CHÍNH
// ------------------------------------------
void loop()
{
  Serial.println("--- Ví dụ điều chỉnh độ sáng riêng lẻ ---");

    // Tắt tất cả LED
    strip.clear();
    strip.show();
    delay(1000);

    uint32_t baseColor = strip.Color(255, 255, 255); // Màu trắng gốc

    // LED 0: 10% độ sáng
    uint32_t color_10 = dimColor(baseColor, 0.1);
    strip.setPixelColor(0, color_10); 
    Serial.println("LED 0: 10%");

    // LED 5: 30% độ sáng
    uint32_t color_30 = dimColor(baseColor, 0.3);
    strip.setPixelColor(5, color_30);
    Serial.println("LED 5: 30%");

    // LED 10: 60% độ sáng
    uint32_t color_60 = dimColor(baseColor, 0.6);
    strip.setPixelColor(10, color_60);
    Serial.println("LED 10: 60%");

    // LED 15: 100% độ sáng
    uint32_t color_100 = dimColor(baseColor, 1.0);
    strip.setPixelColor(15, color_100);
    Serial.println("LED 15: 100%");
    
    // LED 20: Đỏ, 50% độ sáng
    uint32_t redColor = strip.Color(255, 0, 0);
    uint32_t dimRed = dimColor(redColor, 0.5);
    strip.setPixelColor(20, dimRed);
    Serial.println("LED 20: Đỏ, 50%");

    // Cập nhật tất cả LED một lần
    strip.show(); 
    delay(3000);
}