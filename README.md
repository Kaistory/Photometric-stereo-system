# 🌟 Hệ Thống Điều Khiển LED WS2812 (Photometric Stereo System)

Dự án này là module điều khiển ánh sáng cho hệ thống **Photometric Stereo**, cho phép điều khiển dải/vòng LED WS2812 một cách linh hoạt và độc lập từng bóng LED thông qua vi điều khiển **ESP32**. 

Hệ thống hỗ trợ 2 phương thức kết nối: **Bluetooth Low Energy (BLE)** và **MQTT (qua WiFi)**, đi kèm với một ứng dụng Android hiện đại (Jetpack Compose) để tùy chỉnh màu sắc, độ sáng và hiệu ứng.

---

## 📦 Thành phần dự án (Cấu trúc mã nguồn)

Dự án bao gồm 3 tệp mã nguồn chính:

1. **`Code-handware/BLE-40LED/BLE-40LED.ino`**: 
   - Firmware ESP32 điều khiển **40 LED** qua giao thức **BLE**.
   - Có tích hợp màn hình **OLED SSD1306** để hiển thị trạng thái.
   - Hỗ trợ các hiệu ứng khởi động (Rainbow, Police Strobe).
2. **`Code-handware/control_ws2812_mqtt/control_ws2812_mqtt.ino`**: 
   - Firmware ESP32 điều khiển **24 LED** qua giao thức **MQTT** (WiFi).
   - Kết nối với broker HiveMQ Cloud.
   - Lắng nghe topic `esp32/controlled/+` để nhận lệnh đổi màu/độ sáng cho từng LED.
3. **`ledcontrol-master/app/src/main/java/.../MainActivity.kt`**: 
   - Ứng dụng Android viết bằng **Kotlin & Jetpack Compose**.
   - Kết nối với ESP32 thông qua MQTT broker.
   - Giao diện người dùng trực quan, thiết kế theo dạng vòng tròn LED (Circular Layout) đồng bộ với phần cứng thực tế.

---

## 🛠️ Yêu cầu phần cứng

- Vi điều khiển: **ESP32** (hỗ trợ cả WiFi và BLE)
- Đèn LED: Vòng/dải LED **WS2812 (NeoPixel)** (24 bóng hoặc 40 bóng)
- Màn hình: **OLED SSD1306 128x64** (Giao tiếp I2C) *(Chỉ cần cho phiên bản BLE)*
- Nguồn điện: Nguồn 5V DC đủ dòng (khuyến cáo >= 2A để chạy LED ở độ sáng cao).

---

## ✨ Tính năng nổi bật của Ứng dụng Android

- **Giao diện trực quan (Glassmorphism & Neon Glow):** Vòng LED giả lập trên app hiển thị chính xác trạng thái màu sắc và vị trí của từng bóng LED thực tế.
- **Điều khiển độc lập & Toàn cục:** Có thể chỉnh màu/độ sáng cho từng bóng LED riêng lẻ, hoặc dùng chế độ *Global Brightness* để áp dụng cho toàn bộ vòng LED.
- **Thanh trượt Gradient:** Tùy chỉnh trực tiếp các giá trị R (Đỏ), G (Xanh lá), B (Xanh dương) và Độ sáng.
- **Presets & Palettes:**
  - Đi kèm với 10 hiệu ứng màu có sẵn: *Rainbow, Ocean Wave, Fire, Aurora, Sunset, v.v.*
  - Tính năng **Save Preset**: Cho phép người dùng tự lưu lại cấu hình màu sắc yêu thích và gán icon tùy chỉnh. Dữ liệu được lưu trữ local.
- **Kết nối MQTT thời gian thực:** Giao tiếp nhanh chóng, độ trễ thấp thông qua HiveMQ bằng thư viện Paho MQTT.

---

## 🚀 Hướng dẫn Cài đặt & Sử dụng

### 1. Nạp Firmware cho ESP32 (Arduino IDE)

**Chuẩn bị thư viện (Libraries):**
Cài đặt các thư viện sau qua Library Manager trong Arduino IDE:
- `Adafruit NeoPixel`
- `PubSubClient` (Cho bản MQTT)
- `Adafruit GFX Library` & `Adafruit SSD1306` (Cho bản BLE)

**Đối với bản MQTT (`control_ws2812_mqtt.ino`):**
1. Mở file mã nguồn.
2. Cập nhật thông tin WiFi của bạn:
   ```cpp
   const char* ssid = "TÊN_WIFI_CỦA_BẠN";
   const char* password = "MẬT_KHẨU_WIFI";
