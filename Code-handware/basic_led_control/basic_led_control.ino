#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>

// --- CẤU HÌNH WIFI ---
const char* ssid = "TP-Link_847E";      // Tên Wifi
const char* password = "39926227";       // Thay mật khẩu WiFi vào đây

// --- CẤU HÌNH CHÂN LED ---
const int redPin = D0;
const int greenPin = D1;
const int bluePin = D2;

ESP8266WebServer server(80); // Khởi tạo server ở cổng 80

// --- GIAO DIỆN WEBSITE (HTML + JAVASCRIPT) ---
// Đoạn này tạo ra một bảng chọn màu (Color Picker)
const char MAIN_page[] PROGMEM = R"=====(
<!DOCTYPE html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ESP8266 LED Control</title>
  <style>
    body { font-family: Arial; text-align: center; margin-top: 50px; background-color: #f2f2f2;}
    h1 { color: #333; }
    input[type="color"] { width: 100px; height: 100px; border: none; cursor: pointer; }
    p { color: #666; }
  </style>
</head>
<body>
  <h1>Dieu khien Mau LED</h1>
  <p>Chon mau ben duoi:</p>
  <input type="color" id="colorPicker" value="#ffffff" oninput="sendColor(this.value)">
  
  <script>
    function sendColor(hex) {
      // Gửi mã màu Hex (ví dụ #ff0000) về cho ESP8266
      var xhr = new XMLHttpRequest();
      // Bỏ dấu # ở đầu chuỗi hex
      var colorCode = hex.replace('#', ''); 
      xhr.open("GET", "/set?color=" + colorCode, true);
      xhr.send();
    }
  </script>
</body>
</html>
)=====";

// --- HÀM CÀI ĐẶT ---
void setup() {
  Serial.begin(115200);
  
  // Cài đặt chân LED là Output
  pinMode(redPin, OUTPUT);
  pinMode(greenPin, OUTPUT);
  pinMode(bluePin, OUTPUT);
  
  // Tắt đèn ban đầu
  analogWrite(redPin, 0);
  analogWrite(greenPin, 0);
  analogWrite(bluePin, 0);

  // Kết nối WiFi
  WiFi.begin(ssid, password);
  Serial.print("Dang ket noi WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.print("Da ket noi! Dia chi IP cua Web: ");
  Serial.println(WiFi.localIP()); // In địa chỉ IP ra Serial Monitor

  // Cài đặt các đường dẫn cho Server
  server.on("/", handleRoot);      // Khi vào trang chủ
  server.on("/set", handleSetColor); // Khi chỉnh màu

  server.begin();
  Serial.println("Web Server da san sang!");
}

// --- VÒNG LẶP CHÍNH ---
void loop() {
  server.handleClient(); // Luôn luôn lắng nghe yêu cầu từ người dùng
}

// --- CÁC HÀM XỬ LÝ ---

// 1. Hàm hiển thị giao diện web
void handleRoot() {
  server.send(200, "text/html", MAIN_page);
}

// 2. Hàm nhận màu từ web và đổi màu đèn
void handleSetColor() {
  if (server.hasArg("color")) {
    String hexColor = server.arg("color"); // Nhận chuỗi hex (ví dụ: ff0000)
    
    // Chuyển đổi từ Hex sang RGB
    long number = strtol(hexColor.c_str(), NULL, 16);
    int r = number >> 16;
    int g = number >> 8 & 0xFF;
    int b = number & 0xFF;

    // Điều khiển LED (Lưu ý: analogWrite ESP8266 mặc định 0-1023, code cũ bạn dùng 255)
    // Tôi sẽ map giá trị từ 0-255 lên 0-1023 để sáng tốt hơn
    analogWrite(redPin, map(r, 0, 255, 0, 1023));
    analogWrite(greenPin, map(g, 0, 255, 0, 1023));
    analogWrite(bluePin, map(b, 0, 255, 0, 1023));

    Serial.print("Da doi mau: R="); Serial.print(r);
    Serial.print(" G="); Serial.print(g);
    Serial.print(" B="); Serial.println(b);
    
    server.send(200, "text/plain", "OK");
  }
}