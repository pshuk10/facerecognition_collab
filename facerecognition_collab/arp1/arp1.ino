#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>  // For parsing Flask JSON response

// ================== CONFIGURATION ==================
const char* ssid = "vivo Y21G";
const char* password = "00000000";
const char* serverURL = "http://10.129.55.252:8000/upload"; // Flask server IP

#define RELAY_PIN 4           // Relay control pin (use GPIO12 or GPIO14)
#define RELAY_ON HIGH
#define RELAY_OFF LOW
#define RELAY_ACTIVE_TIME 2000 // ms to keep relay ON

#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

#define FLASH_GPIO_NUM     4
bool useFlash = false;

// ================== WIFI ==================
void connectToWiFi() {
  Serial.printf("Connecting to %s", ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n✅ WiFi Connected!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());
}

// ================== CAMERA ==================
void setupCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  config.frame_size = FRAMESIZE_QVGA;
  config.jpeg_quality = 5;
  config.fb_count = 2;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("❌ Camera init failed with error 0x%x\n", err);
  } else {
    Serial.println("📸 Camera initialized successfully!");
  }
}

// ================== RELAY CONTROL ==================
void triggerRelay() {
  Serial.println("🔓 Relay ON (door unlock)");
  digitalWrite(RELAY_PIN, RELAY_ON);
  delay(RELAY_ACTIVE_TIME);
  digitalWrite(RELAY_PIN, RELAY_OFF);
  Serial.println("🔒 Relay OFF (door lock)");
}

// ================== CAPTURE & UPLOAD ==================
void captureAndUpload() {
  if (WiFi.status() != WL_CONNECTED) {
    connectToWiFi();
  }

  if (useFlash) {
    digitalWrite(FLASH_GPIO_NUM, HIGH);
    delay(500);
  }

  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("❌ Camera capture failed!");
    if (useFlash) digitalWrite(FLASH_GPIO_NUM, LOW);
    delay(2000);
    return;
  }

  if (useFlash) digitalWrite(FLASH_GPIO_NUM, LOW);

  Serial.printf("📤 Captured %d bytes. Uploading...\n", fb->len);

  String boundary = "----ESP32Boundary";
  String bodyStart = "--" + boundary + "\r\n";
  bodyStart += "Content-Disposition: form-data; name=\"image\"; filename=\"capture.jpg\"\r\n";
  bodyStart += "Content-Type: image/jpeg\r\n\r\n";
  String bodyEnd = "\r\n--" + boundary + "--\r\n";

  int totalLen = bodyStart.length() + fb->len + bodyEnd.length();
  uint8_t* payload = (uint8_t*)malloc(totalLen);
  if (!payload) {
    Serial.println("Memory allocation failed!");
    esp_camera_fb_return(fb);
    return;
  }

  memcpy(payload, bodyStart.c_str(), bodyStart.length());
  memcpy(payload + bodyStart.length(), fb->buf, fb->len);
  memcpy(payload + bodyStart.length() + fb->len, bodyEnd.c_str(), bodyEnd.length());

  HTTPClient http;
  http.begin(serverURL);
  http.addHeader("Content-Type", "multipart/form-data; boundary=" + boundary);

  int code = http.POST(payload, totalLen);
  if (code > 0) {
    String response = http.getString();
    Serial.printf("✅ Server responded: %d\n", code);
    Serial.println(response);

    // Parse JSON to check for "authorised"
    StaticJsonDocument<256> doc;
    DeserializationError error = deserializeJson(doc, response);

    if (!error && doc["status"] == "authorised") {
      triggerRelay();
    } else {
      Serial.println("❌ Not authorised or face not recognised.");
    }

  } else {
    Serial.printf("Upload failed: %s\n", http.errorToString(code).c_str());
  }

  free(payload);
  esp_camera_fb_return(fb);
  http.end();
}

// ================== SETUP ==================
void setup() {
  Serial.begin(115200);
  Serial.println("\nBooting ESP32-CAM Face Recognition Relay System...");
  
  pinMode(FLASH_GPIO_NUM, OUTPUT);
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, RELAY_OFF);

  connectToWiFi();
  setupCamera();
}

// ================== LOOP ==================
void loop() {
  captureAndUpload();
  delay(8000); // Wait before next frame
}
