// #include "nn_weights.h"

// #include <DHT.h>
// #include <Wire.h>
// #include <Adafruit_BMP085.h>
// #include <Adafruit_GFX.h>
// #include <Adafruit_SSD1306.h>
// #include <BLEDevice.h>
// #include <BLEServer.h>
// #include <BLEUtils.h>
// #include <BLE2902.h>
// #include <Preferences.h>
// #include <math.h>
// #include <esp_now.h>
// #include <WiFi.h>

// struct WeatherResult {
//     const char* name;
//     const char* detail;
//     int severity;
//     const char* icon;
// };

// struct TelemetryRecord {
//     float curTemp, curHum, curPres, curLux;
//     float predTemp, predHum, predPres, predLux;
//     float pressureTrend;
//     int weatherSev;
//     char weatherName[20];
//     uint32_t wakeNum;
// };

// uint8_t broadcastAddress[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

// void OnDataSent(const wifi_tx_info_t *info, esp_now_send_status_t status) {
//     Serial.print("\r\nLast Packet Send Status:\t");
//     Serial.println(status == ESP_NOW_SEND_SUCCESS ? "Delivery Success" : "Delivery Fail");
// }

// #define NN_HIDDEN1 16
// #define NN_HIDDEN2 8
// #define NN_FEATURES 4

// #define SCREEN_WIDTH 128
// #define SCREEN_HEIGHT 64
// #define OLED_RESET -1
// #define SCREEN_ADDRESS 0x3C

// #define SENSOR_POWER_PIN 15
// #define DHT_PIN 4
// #define LIGHT_PIN 34
// #define CONFIRM_LED_PIN   14

// #define SAMPLE_INTERVAL_MS 5000
// #define NUM_SAMPLES 12
// #define BLE_WINDOW_MS 30000

// #define SLEEP_SEVERE 1
// #define SLEEP_MILD 1
// #define SLEEP_CLEAR 1

// #define SERVICE_UUID             "e3a1f0b0-1234-5678-abcd-000000000001"
// #define CHAR_CURRENT_UUID        "e3a1f0b0-1234-5678-abcd-000000000002"
// #define CHAR_PREDICTED_UUID      "e3a1f0b0-1234-5678-abcd-000000000003"
// #define CHAR_WEATHER_STATUS_UUID "e3a1f0b0-1234-5678-abcd-000000000004"
// #define CHAR_ALERT_UUID          "e3a1f0b0-1234-5678-abcd-000000000005"
// #define CHAR_TREND_UUID          "e3a1f0b0-1234-5678-abcd-000000000006"
// #define CHAR_HISTORY_UUID        "e3a1f0b0-1234-5678-abcd-000000000007"

// #define HISTORY_SLOTS 32
// #define NVS_NAMESPACE "wmind"

// struct KalmanFilter {
//     float Q;
//     float R;
//     float x;
//     float P;
//     bool init;

//     void reset(float q, float r) {
//         Q = q; R = r; x = 0.0f; P = 1.0f; init = false;
//     }

//     float update(float z) {
//         if (!init) { x = z; P = 1.0f; init = true; return x; }
//         P = P + Q;
//         float K = P / (P + R);
//         x = x + K * (z - x);
//         P = (1.0f - K) * P;
//         return x;
//     }
// };

// struct PressureTrend {
//     static constexpr int WINDOW = 8;
//     float samples[WINDOW];
//     int count = 0;
//     int head = 0;

//     void push(float p) {
//         samples[head % WINDOW] = p;
//         head++;
//         if (count < WINDOW) count++;
//     }

//     float slopePerMin() const {
//         if (count < 2) return 0.0f;
//         int n = count;
//         float sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
//         for (int i = 0; i < n; i++) {
//             float x = (float)i;
//             float y = samples[(head - n + i + WINDOW * 100) % WINDOW];
//             sumX += x; sumY += y;
//             sumXY += x * y; sumX2 += x * x;
//         }
//         float denom = n * sumX2 - sumX * sumX;
//         if (fabsf(denom) < 1e-6f) return 0.0f;
//         float slope = (n * sumXY - sumX * sumY) / denom;
//         return slope * (60.0f / (SAMPLE_INTERVAL_MS / 1000.0f));
//     }

//     const char* label() const {
//         float s = slopePerMin();
//         if (s > 1.5f) return "RISING FAST";
//         if (s > 0.4f) return "RISING";
//         if (s < -1.5f) return "FALLING FAST";
//         if (s < -0.4f) return "FALLING";
//         return "STEADY";
//     }
// };



// DHT dht(DHT_PIN, DHT11);
// Adafruit_BMP085 bmp;
// Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
// Preferences prefs;

// KalmanFilter kf[4];
// PressureTrend pressTrend;

// RTC_DATA_ATTR int wakeCount = 0;
// RTC_DATA_ATTR char prevWeatherName[20] = "UNKNOWN";
// RTC_DATA_ATTR int prevSeverity = -1;
// RTC_DATA_ATTR float prevPressureMb = 0.0f;
// RTC_DATA_ATTR uint8_t historyHead = 0;

// float sensorBuffer[NN_LOOKBACK][NN_FEATURES];
// float inputVec[NN_INPUT_DIM];
// float hidden1Buf[NN_HIDDEN1];
// float hidden2Buf[NN_HIDDEN2];
// float outputBuf[NN_OUTPUT_DIM];

// float curTemp, curHum, curPres, curLux;
// float predTemp = 0.0f, predHum = 0.0f, predPres = 0.0f, predLux = 0.0f;
// float pressureSlopePaMin = 0.0f;

// const char* weatherName = "UNKNOWN";
// const char* weatherDetail = "";
// const char* weatherIcon = "?";
// int weatherSev = 0;
// bool alertEscalated = false;

// static inline float pgm_float(const float* addr) {
//     float v; memcpy_P(&v, addr, sizeof(float)); return v;
// }
// static inline float relu(float x) { return x > 0.0f ? x : 0.0f; }
// static inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

// void denseForward(const float* in, int inN, float* out, int outN,
//                   const float* W, const float* B, char act) {
//     for (int j = 0; j < outN; j++) {
//         float sum = pgm_float(&B[j]);
//         for (int i = 0; i < inN; i++) {
//             sum += in[i] * pgm_float(&W[i * outN + j]);
//         }
//         switch (act) {
//             case 'r': sum = relu(sum); break;
//             case 's': sum = sigmoidf(sum); break;
//             default: break;
//         }
//         out[j] = sum;
//     }
// }

// void nnPredict(float* in, float* out) {
//     denseForward(in, NN_INPUT_DIM, hidden1Buf, NN_HIDDEN1, W_HIDDEN1, B_HIDDEN1, 'r');
//     denseForward(hidden1Buf, NN_HIDDEN1, hidden2Buf, NN_HIDDEN2, W_HIDDEN2, B_HIDDEN2, 'r');
//     denseForward(hidden2Buf, NN_HIDDEN2, out, NN_OUTPUT_DIM, W_OUTPUT, B_OUTPUT, 's');
// }

// float normalizeVal(float raw, int i) {
//     return (raw - pgm_float(&FEAT_MIN[i])) / pgm_float(&FEAT_RANGE[i]);
// }
// float denormalizeVal(float n, int i) {
//     return n * pgm_float(&FEAT_RANGE[i]) + pgm_float(&FEAT_MIN[i]);
// }

// float iqrMean(float* vals, int n) {
//     float sorted[NUM_SAMPLES];
//     memcpy(sorted, vals, n * sizeof(float));
//     for (int i = 0; i < n - 1; i++)
//         for (int j = i + 1; j < n; j++)
//             if (sorted[j] < sorted[i]) { float tmp = sorted[i]; sorted[i] = sorted[j]; sorted[j] = tmp; }

//     float q1 = sorted[n / 4];
//     float q3 = sorted[(3 * n) / 4];
//     float iqr = q3 - q1;
//     float lo = q1 - 1.5f * iqr;
//     float hi = q3 + 1.5f * iqr;

//     float sum = 0.0f; int cnt = 0;
//     for (int i = 0; i < n; i++)
//         if (vals[i] >= lo && vals[i] <= hi) { sum += vals[i]; cnt++; }
//     return cnt > 0 ? sum / cnt : 0.0f;
// }

// void applyPressureCorrection(float& predPres, float slopePaMin) {
//     predPres += slopePaMin * 30.0f;
// }

// WeatherResult classifyWeather(float tempC, float humPct,
//                               float pressPa, float lux,
//                               float slopePaMin) {
//     float pressMb = pressPa / 100.0f;
//     bool rapidDrop = slopePaMin < -2.0f;

//     if (tempC > 26.7f && pressMb < 980.0f && humPct >= 90.0f && lux < 50.0f)
//         return {"HURRICANE", "Extreme low pressure, warm & humid", 3, "HUR"};
//     if (tempC < -7.0f && pressMb < 995.0f && humPct > 80.0f && lux < 100.0f)
//         return {"BLIZZARD", "Extreme cold + low pressure", 3, "BLZ"};
//     if ((rapidDrop || pressMb < 1000.0f) && tempC > 18.0f && humPct > 70.0f)
//         return {"THUNDERSTORM", "Low pressure, warm & humid — storm approach", 3, "THN"};

//     if (tempC < 0.0f && pressMb < 1010.0f && humPct > 70.0f)
//         return {"SNOW", "Sub-zero + humid + low pressure", 2, "SNW"};
//     if (tempC >= 4.4f && tempC <= 26.7f && pressMb >= 990.0f && pressMb <= 1005.0f
//         && humPct >= 60.0f && lux >= 500.0f && lux <= 2000.0f)
//         return {"RAIN", "Low pressure, humid & overcast", 2, "RAN"};
//     if (tempC > 38.0f && humPct < 40.0f && lux > 500.0f)
//         return {"HEAT WAVE", "Extreme heat, dry & bright", 2, "HT!"};
//     if (rapidDrop && pressMb < 1010.0f && lux < 100.0f)
//         return {"STORM WATCH", "Rapid pressure drop detected", 2, "WRN"};

//     if (humPct >= 95.0f && pressMb > 1013.0f && lux < 100.0f)
//         return {"FOG", "Saturated humidity, low visibility", 1, "FOG"};
//     if (pressMb < 1010.0f && humPct > 60.0f)
//         return {"OVERCAST", "Low pressure, elevated humidity", 1, "OVC"};
//     if (humPct >= 70.0f && humPct < 95.0f && pressMb >= 1008.0f)
//         return {"CLOUDY", "Partly cloudy conditions", 1, "CLD"};

//     return {"CLEAR", "No severe weather indicators", 0, "CLR"};
// }

// bool readSensors(float* temp, float* hum, float* pres, float* lux) {
//     float t = dht.readTemperature();
//     float h = dht.readHumidity();
//     if (isnan(t) || isnan(h)) return false;
//     if (t < -40.0f || t > 80.0f || h < 0.0f || h > 100.0f) return false;
//     *temp = kf[0].update(t);
//     *hum  = kf[1].update(h);
//     *pres = kf[2].update((float)bmp.readSealevelPressure(515));
//     float rawADC = (float)analogRead(LIGHT_PIN);
//     *lux  = kf[3].update(rawADC * (632.0f / 4095.0f));
//     return true;
// }

// void drawProgressBar(int x, int y, int w, int h, int pct) {
//     display.drawRect(x, y, w, h, SSD1306_WHITE);
//     int fill = (w - 2) * pct / 100;
//     if (fill > 0) display.fillRect(x + 1, y + 1, fill, h - 2, SSD1306_WHITE);
// }

// void renderCollectingScreen(int sample, int total, float t, float h, float p, float l) {
//     display.clearDisplay();
//     display.setTextSize(1);
//     display.setTextColor(SSD1306_WHITE);

//     display.setCursor(0, 0);
//     display.print("WeatherMind");
//     display.setCursor(80, 0);
//     display.print("SAMPLING"); 

//     int pct = sample * 100 / total;
//     drawProgressBar(0, 11, 128, 6, pct);

    
//     display.setCursor(0, 24);
//     display.printf("T:%.1fC", t);
//     display.setCursor(66, 24);
//     display.printf("H:%.0f%%", h);

//     display.setCursor(0, 34);
//     display.printf("P:%.0fPa", p);
//     display.setCursor(66, 34);
//     display.printf("L:%.0f", l);

//     display.drawLine(0, 50, 127, 50, SSD1306_WHITE); 
    
//     display.setCursor(0, 55);
//     display.printf("Progress: %d/%d", sample, total);

//     display.display();
// }

// void drawDangerTriangle(int x, int y) {
//     display.drawLine(x + 5, y,     x,      y + 9, SSD1306_WHITE);
//     display.drawLine(x + 5, y,     x + 10, y + 9, SSD1306_WHITE);
//     display.drawLine(x,     y + 9, x + 10, y + 9, SSD1306_WHITE);
//     display.drawLine(x + 5, y + 3, x + 5, y + 6, SSD1306_WHITE);
//     display.drawPixel(x + 5, y + 8, SSD1306_WHITE);
// }

// void renderResultScreen() {
//     display.clearDisplay();
//     display.setTextSize(1);
//     display.setTextColor(SSD1306_WHITE);

//     // Current T + H on one row
//     display.setCursor(0, 0);
//     display.printf("T:%.1fC", curTemp);
//     display.setCursor(66, 0);
//     display.printf("H:%.0f%%", curHum);

//     display.setCursor(0, 10);
//     display.printf("P:%.0fPa", curPres);

//     display.drawLine(0, 23, 127, 23, SSD1306_WHITE);
//     display.fillRect(40, 20, 48, 7, SSD1306_BLACK);
//     display.setCursor(43, 20);
//     display.print("30 min");

//     display.setCursor(0, 30);
//     display.printf("T:%.1fC", predTemp);
//     display.setCursor(66, 30);
//     display.printf("H:%.0f%%", predHum);

//     display.setCursor(0, 40);
//     display.printf("P:%.0fPa", predPres);

//     display.setCursor(0, 54);
//     display.print(">> ");
//     display.print(weatherName);

//     if (weatherSev > 0) {
//         drawDangerTriangle(115, 54);
//     }

//     display.display();
// }

// void saveToFlash(const TelemetryRecord& rec) {
//     prefs.begin(NVS_NAMESPACE, false);
//     char key[12];
//     snprintf(key, sizeof(key), "rec%02u", (unsigned)(historyHead % HISTORY_SLOTS));
//     prefs.putBytes(key, &rec, sizeof(TelemetryRecord));
//     historyHead++;
//     prefs.putUInt("head", historyHead);
//     prefs.end();
// }

// int readHistory(TelemetryRecord* buf, int maxN) {
//     prefs.begin(NVS_NAMESPACE, true);
//     uint32_t head = prefs.getUInt("head", 0);
//     int cnt = min((int)head, min(maxN, HISTORY_SLOTS));
//     for (int i = 0; i < cnt; i++) {
//         char key[12];
//         uint32_t slot = (head - cnt + i) % HISTORY_SLOTS;
//         snprintf(key, sizeof(key), "rec%02u", (unsigned)slot);
//         prefs.getBytes(key, &buf[i], sizeof(TelemetryRecord));
//     }
//     prefs.end();
//     return cnt;
// }

// void runBLE() {
//     Serial.println("[BLE] Starting GATT server...");
//     BLEDevice::init("WeatherMind");
//     BLEServer* server = BLEDevice::createServer();
//     BLEService* service = server->createService(SERVICE_UUID);

//     auto addChar = [&](const char* uuid, const char* value) -> BLECharacteristic* {
//         BLECharacteristic* ch = service->createCharacteristic(
//             uuid, BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY);
//         ch->addDescriptor(new BLE2902());
//         ch->setValue(value);
//         return ch;
//     };

//     char curStr[64];
//     snprintf(curStr, sizeof(curStr), "%.1f,%.0f,%.0f,%.0f",
//              curTemp, curHum, curPres, curLux);
//     addChar(CHAR_CURRENT_UUID, curStr);

//     char predStr[64];
//     snprintf(predStr, sizeof(predStr), "%.1f,%.0f,%.0f,%.0f",
//              predTemp, predHum, predPres, curLux);
//     addChar(CHAR_PREDICTED_UUID, predStr);

//     char wStr[96];
//     snprintf(wStr, sizeof(wStr), "%s|%d|%s", weatherName, weatherSev, weatherDetail);
//     addChar(CHAR_WEATHER_STATUS_UUID, wStr);

//     char alertStr[4];
//     snprintf(alertStr, sizeof(alertStr), "%d", alertEscalated ? 1 : 0);
//     addChar(CHAR_ALERT_UUID, alertStr);

//     char trendStr[32];
//     snprintf(trendStr, sizeof(trendStr), "%s|%.2f", pressTrend.label(), pressureSlopePaMin);
//     addChar(CHAR_TREND_UUID, trendStr);

//     TelemetryRecord hist[8];
//     int hcnt = readHistory(hist, 8);
//     String histStr = "";
//     for (int i = 0; i < hcnt; i++) {
//         char line[64];
//         snprintf(line, sizeof(line), "%.1f,%.0f,%.1f,%s;",
//                  hist[i].curTemp, hist[i].curHum,
//                  hist[i].curPres / 100.0f, hist[i].weatherName);
//         histStr += line;
//     }
//     BLECharacteristic* histCh = service->createCharacteristic(
//         CHAR_HISTORY_UUID, BLECharacteristic::PROPERTY_READ);
//     histCh->setValue(histStr.c_str());

//     service->start();

//     BLEAdvertising* adv = BLEDevice::getAdvertising();
//     adv->addServiceUUID(SERVICE_UUID);
//     adv->setScanResponse(true);
//     adv->setMinPreferred(0x06);
//     BLEDevice::startAdvertising();

//     Serial.printf("[BLE] Advertising for %d ms...\n", BLE_WINDOW_MS);
//     delay(BLE_WINDOW_MS);

//     BLEDevice::stopAdvertising();
//     BLEDevice::deinit(true);
//     Serial.println("[BLE] Done.");
// }

// void logSection(const char* title) {
//     Serial.printf("\n┌─────────────────────────────────────────\n│ %s\n└─────────────────────────────────────────\n", title);
// }

// void sendEspNow(const TelemetryRecord& data) {
//     pinMode(CONFIRM_LED_PIN, OUTPUT);
//     for(int i = 0; i < 5; i++) {
//         digitalWrite(CONFIRM_LED_PIN, HIGH);  
//         delay(150);
//         digitalWrite(CONFIRM_LED_PIN, LOW);
//         delay(150);
//     }
//     WiFi.mode(WIFI_STA);
//     if (esp_now_init() != ESP_OK) {
//         return;
//     }

//     esp_now_register_send_cb(OnDataSent);

//     esp_now_peer_info_t peerInfo = {};
//     memcpy(peerInfo.peer_addr, broadcastAddress, 6);
//     peerInfo.encrypt = false;

//     if (esp_now_add_peer(&peerInfo) == ESP_OK) {
//         esp_now_send(broadcastAddress, (uint8_t*)&data, sizeof(data));
//     }

//     delay(200);
//     digitalWrite(CONFIRM_LED_PIN, LOW);

//     esp_now_deinit();
//     WiFi.mode(WIFI_OFF);
// }

// void renderTransmissionScreen(bool sent) {
//     display.clearDisplay();
//     display.setTextSize(1);
//     display.setTextColor(SSD1306_WHITE);

//     display.setCursor(0, 0);
//     display.print("WeatherMind");

//     display.drawLine(0, 12, 127, 12, SSD1306_WHITE);

//     if (sent) {
//         display.setCursor(0, 25);
//         display.print("Weather: ALERT");

//         display.setCursor(0, 40);
//         display.print("Sending to modules...");
//     } else {
//         display.setCursor(0, 25);
//         display.print("Weather: CLEAR");

//         display.setCursor(0, 40);
//         display.print("Needs no transmission");
//     }

//     display.display();
// }

// void setup() {
//     Serial.begin(115200);
//     wakeCount++;

//     logSection("WeatherMind Pro — Wake");
//     Serial.printf("  Wake #%d\n", wakeCount);

//     kf[0].reset(0.01f, 0.5f);
//     kf[1].reset(0.01f, 1.0f);
//     kf[2].reset(0.5f, 5.0f);
//     kf[3].reset(5.0f, 20.0f);

//     pinMode(SENSOR_POWER_PIN, OUTPUT);
//     digitalWrite(SENSOR_POWER_PIN, HIGH);
//     delay(3000);

//     Wire.begin(21, 22);
//     dht.begin();
//     analogReadResolution(12);
//     pinMode(LIGHT_PIN, INPUT);

//     if (!bmp.begin()) {
//         Serial.println("[ERROR] BMP180 not found — halting");
//         if (!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
//             while (true) delay(1000);
//         }
//         display.clearDisplay();
//         display.setTextSize(1);
//         display.setTextColor(SSD1306_WHITE);
//         display.setCursor(0, 20);
//         display.println("BMP180 ERROR");
//         display.display();
//         while (true) delay(1000);
//     }

//     if (!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
//         Serial.println("[ERROR] OLED not found — halting");
//         while (true) delay(1000);
//     }

//     display.clearDisplay();
//     display.display();
//     delay(200);

//     display.fillRect(0, 0, 128, 11, SSD1306_WHITE);
//     display.setTextColor(SSD1306_BLACK);
//     display.setTextSize(1);
//     display.setCursor(32, 2); 
//     display.print("WEATHERMIND"); 

//     display.setTextColor(SSD1306_WHITE);
    
//     display.setCursor(10, 28); 
//     display.print("by Shadow Mechanics");
    
//     display.drawLine(20, 39, 108, 39, SSD1306_WHITE);

//     display.setCursor(42, 45);
//     display.printf("Session #%d", wakeCount);

//     display.display();
//     delay(2500);

//     dht.readTemperature();
//     dht.readHumidity();
//     delay(2500);

//     logSection("Data Collection");

//     float rawT[NUM_SAMPLES], rawH[NUM_SAMPLES], rawP[NUM_SAMPLES], rawL[NUM_SAMPLES];
//     int samplesOk = 0;

//     for (int s = 0; s < NUM_SAMPLES; s++) {
//         float t, h, p, l;
//         bool ok = readSensors(&t, &h, &p, &l);

//         if (ok) {
//             sensorBuffer[s][0] = normalizeVal(t, 0);
//             sensorBuffer[s][1] = normalizeVal(h, 1);
//             sensorBuffer[s][2] = normalizeVal(p, 2);
//             sensorBuffer[s][3] = normalizeVal(l, 3);

//             rawT[s] = t; rawH[s] = h; rawP[s] = p; rawL[s] = l;
//             pressTrend.push(p);
//             curTemp = t; curHum = h; curPres = p; curLux = l;
//             samplesOk++;
//             Serial.printf("  [%2d/%d] T:%5.1fC H:%4.1f%% P:%7.0fPa L:%5.0f  [KF]\n",
//                           s + 1, NUM_SAMPLES, t, h, p, l);
//         } else {
//             if (s > 0) {
//                 for (int f = 0; f < NN_FEATURES; f++)
//                     sensorBuffer[s][f] = sensorBuffer[s - 1][f];
//                 rawT[s] = curTemp; rawH[s] = curHum;
//                 rawP[s] = curPres; rawL[s] = curLux;
//             }
//             Serial.printf("  [%2d/%d] DHT read failed — using KF estimate\n", s + 1, NUM_SAMPLES);
//         }

//         renderCollectingScreen(s + 1, NUM_SAMPLES, curTemp, curHum, curPres, curLux);

//         if (s < NUM_SAMPLES - 1) delay(SAMPLE_INTERVAL_MS);
//     }

//     Serial.printf("  Collected %d/%d valid samples\n", samplesOk, NUM_SAMPLES);

//     if (samplesOk > 2) {
//         curTemp = iqrMean(rawT, samplesOk);
//         curHum = iqrMean(rawH, samplesOk);
//         curPres = iqrMean(rawP, samplesOk);
//         curLux = iqrMean(rawL, samplesOk);
//     }

//     pressureSlopePaMin = pressTrend.slopePerMin();
//     Serial.printf("  Pressure trend: %.2f Pa/min  [%s]\n",
//                   pressureSlopePaMin, pressTrend.label());

//     logSection("Neural Network Inference");

//     if (samplesOk >= NUM_SAMPLES / 2) {
//         for (int s = 0; s < NN_LOOKBACK; s++) {
//             for (int f = 0; f < NN_FEATURES; f++) {
//                 inputVec[s * NN_FEATURES + f] = sensorBuffer[s][f];
//             }
//         }

//         nnPredict(inputVec, outputBuf);

//         Serial.printf("Raw Norm Temp: %.6f\n", outputBuf[0]);
//         Serial.printf("Raw Norm Hum:  %.6f\n", outputBuf[1]);
//         Serial.printf("Raw Norm Pres: %.6f\n", outputBuf[2]);

//         predTemp = denormalizeVal(outputBuf[0], 0);
//         predHum  = denormalizeVal(outputBuf[1], 1);
//         predPres = denormalizeVal(outputBuf[2], 2);
//         predLux = curLux;

//         applyPressureCorrection(predPres, pressureSlopePaMin);

//         predTemp = constrain(predTemp, -50.0f, 80.0f);
//         predHum = constrain(predHum, 0.0f, 100.0f);
//         predPres = constrain(predPres, 85000.0f, 108000.0f);
//         predLux = constrain(predLux, 0.0f, 632.0f);

//         WeatherResult w = classifyWeather(predTemp, predHum, predPres, curLux, pressureSlopePaMin);
//         weatherName = w.name;
//         weatherDetail = w.detail;
//         weatherSev = w.severity;
//         weatherIcon = w.icon;

//         alertEscalated = (weatherSev > prevSeverity && prevSeverity >= 0);
//         strncpy(prevWeatherName, weatherName, sizeof(prevWeatherName) - 1);
//         prevWeatherName[sizeof(prevWeatherName) - 1] = '\0';
//         prevSeverity = weatherSev;
//         prevPressureMb = curPres / 100.0f;

//         static const char* sevLabel[] = {"CLEAR", "MILD", "MODERATE", "SEVERE"};
//         Serial.printf("  Pred   T:%.1fC  H:%.1f%%  P:%.0fPa  L:%.1f\n",
//                       predTemp, predHum, predPres, predLux);
//         Serial.printf("  Result %s  [%s]  Escalated:%s\n",
//                       weatherName, sevLabel[min(weatherSev,3)],
//                       alertEscalated ? "YES" : "no");
//     } else {
//         weatherName = "NO DATA";
//         weatherDetail = "Insufficient samples";
//         weatherSev = 0;
//         Serial.println("  Skipped — too few valid samples");
//     }

//     TelemetryRecord rec;
//     rec.curTemp = curTemp;  rec.curHum = curHum;
//     rec.curPres = curPres;   rec.curLux = curLux;
//     rec.predTemp = predTemp; rec.predHum = predHum;
//     rec.predPres = predPres; rec.predLux = predLux;
//     rec.pressureTrend = pressureSlopePaMin;
//     rec.weatherSev = weatherSev;
//     rec.wakeNum = wakeCount;
//     strncpy(rec.weatherName, weatherName, sizeof(rec.weatherName) - 1);
//     rec.weatherName[sizeof(rec.weatherName) - 1] = '\0';
//     saveToFlash(rec);
//     Serial.printf("  Saved record #%u to NVS flash\n", wakeCount);

//     logSection("ESP-NOW Transmission");
//     bool sent = false;

//     if (weatherSev > 0) {
//         sendEspNow(rec);
//         sent = true;

//     } else {
//         pinMode(CONFIRM_LED_PIN, OUTPUT);
//         for(int i = 0; i < 3; i++) {
//             digitalWrite(CONFIRM_LED_PIN, HIGH);
//             delay(150);
//             digitalWrite(CONFIRM_LED_PIN, LOW);
//             delay(150);
//         }
//         Serial.println("Skipped — weather is CLEAR, no transmission needed");
//         sent = false;
//     }

//     renderTransmissionScreen(sent);

//     delay(6000);

//     renderResultScreen();

//     digitalWrite(SENSOR_POWER_PIN, LOW);

//     logSection("BLE Broadcast");
//     runBLE();
    
//     int sleepMin;
//     switch (weatherSev) {
//         case 3: sleepMin = SLEEP_SEVERE; break;
//         case 2: sleepMin = SLEEP_MILD; break;
//         default: sleepMin = SLEEP_CLEAR; break;
//     }

//     logSection("Deep Sleep");
//     Serial.printf("  Weather severity: %d  ->  sleeping %d min\n", weatherSev, sleepMin);
//     Serial.printf("  OLED will retain last display during sleep.\n");
//     Serial.flush();

//     esp_sleep_enable_timer_wakeup((uint64_t)sleepMin * 60ULL * 1000000ULL);
//     esp_deep_sleep_start();
// }

// void loop() {
// }
/**
 * FarmSentinel — Farm Sensor Node Firmware
 * =========================================
 * Works as either a SENSOR MODULE or HUB MODULE (set IS_HUB = true for hub).
 *
 * Sensors:
 *   - DHT22            → air temperature + humidity (GPIO 4)
 *   - BMP180           → barometric pressure (I2C: SDA=21, SCL=22)
 *   - LDR/TEMT6000     → light / lux (ADC GPIO 34)
 *   - Capacitive soil  → soil moisture % (ADC GPIO 35)
 *   - DS18B20          → soil temperature °C (OneWire GPIO 32)
 *   - Solar + LiPo     → VBAT monitor (ADC GPIO 33)
 *
 * NN: 36→24→12→2 predicts [min_temp_6h, max_temp_6h]
 * Classification (on-device):
 *   FROST_RISK      if pred_min < 2°C AND lux < 1000 (night/dusk)
 *   HEAT_STRESS     if pred_max > 35°C AND soil_moisture < 20%
 *   DROUGHT_STRESS  if soil_moisture < 15% AND pred_max > 30°C
 *   SAFE            otherwise
 *
 * ESP-NOW: broadcasts FarmPacket to all peers every cycle.
 *          Hub listens and aggregates a zone map.
 * BLE:     Hub advertises GATT with full farm map.
 * WiFi:    Hub serves HTTP JSON endpoint for web dashboard.
 * OLED:    128x64 SSD1306 shows current reading + alert status.
 */

#include "nn_weights.h"

#include <DHT.h>
#include <Wire.h>
#include <Adafruit_BMP085.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <Preferences.h>
#include <esp_now.h>
#include <WiFi.h>
#include <WebServer.h>
#include <algorithm>
#include <math.h>

// ====================================================================
//  NODE CONFIGURATION — edit these per module before flashing
// ====================================================================
#define IS_HUB       false          // set true on the hub module only
#define ZONE_ID      1              // 0=Hub, 1..15=field nodes
#define ZONE_NAME    "North Field"  // up to 15 chars, shown on dashboard
#define ELEV_M       100            // elevation in metres for pressure calc

// WiFi credentials (Hub only)
#define WIFI_SSID    "YourFarmSSID"
#define WIFI_PASS    "YourPassword"

// ====================================================================
//  PIN MAP
// ====================================================================
#define DHT_PIN          4
#define DHT_TYPE         DHT22
#define ONE_WIRE_PIN     32   // DS18B20 data line
#define SOIL_ADC_PIN     35   // capacitive soil moisture sensor
#define LIGHT_ADC_PIN    34   // LDR or TEMT6000 light sensor
#define VBAT_ADC_PIN     33   // battery voltage via divider
#define SENSOR_PWR_PIN   15   // mosfet gate: cuts sensor power in sleep
#define STATUS_LED_PIN   14   // status LED

// Capacitive soil moisture calibration (12-bit ADC)
// Measure your sensor in dry air and submerged in water to get these.
#define SOIL_DRY_ADC   3400
#define SOIL_WET_ADC   1100

// ====================================================================
//  TIMING
// ====================================================================
#define SAMPLE_INTERVAL_MS   3000
#define NUM_SAMPLES          6
#define SLEEP_NORMAL_MIN     60
#define SLEEP_ALERT_MIN      10
#define BLE_WINDOW_MS        20000
#define HTTP_IDLE_MS         30000
#define HTTP_ALERT_MS        60000

// ====================================================================
//  FARM ALERT THRESHOLDS
// ====================================================================
#define FROST_TEMP_C       2.0f
#define HEAT_TEMP_C       35.0f
#define DROUGHT_SOIL_PCT  15.0f
#define NIGHT_LUX         1000.0f

// ====================================================================
//  BLE UUIDs
// ====================================================================
#define SERVICE_UUID        "a1b2c3d4-0001-0001-0001-000000000001"
#define CHAR_FARM_MAP_UUID  "a1b2c3d4-0001-0001-0001-000000000002"
#define CHAR_ALERTS_UUID    "a1b2c3d4-0001-0001-0001-000000000003"
#define CHAR_HUB_UUID       "a1b2c3d4-0001-0001-0001-000000000004"

// ====================================================================
//  OLED
// ====================================================================
#define SCREEN_W   128
#define SCREEN_H   64
#define OLED_RST   -1
#define OLED_ADDR  0x3C

// ====================================================================
//  ESP-NOW DATA PACKET
// ====================================================================
struct __attribute__((packed)) FarmPacket {
    uint8_t  zone_id;
    char     zone_name[16];
    float    air_temp;          // deg C
    float    humidity;          // %
    float    pressure;          // Pa
    float    lux;               // lux
    float    soil_pct;          // 0-100%
    float    soil_temp;         // deg C
    float    pred_min;          // 6h predicted min temp
    float    pred_max;          // 6h predicted max temp
    uint8_t  alert;             // 0=SAFE 1=FROST 2=HEAT 3=DROUGHT
    float    battery_pct;
    uint32_t uptime_s;
};

// ====================================================================
//  ZONE MAP (Hub)
// ====================================================================
#define MAX_ZONES 16
static FarmPacket  zoneMap[MAX_ZONES];
static bool        zoneActive[MAX_ZONES];
static uint32_t    zoneLastSeen[MAX_ZONES];

// ====================================================================
//  HARDWARE OBJECTS
// ====================================================================
DHT              dht(DHT_PIN, DHT_TYPE);
Adafruit_BMP085  bmp;
Adafruit_SSD1306 oled(SCREEN_W, SCREEN_H, &Wire, OLED_RST);
OneWire          ow(ONE_WIRE_PIN);
DallasTemperature ds18(&ow);
Preferences      prefs;
WebServer        http(80);

// ====================================================================
//  PERSISTENT STATE (survives deep sleep via RTC memory)
// ====================================================================
RTC_DATA_ATTR uint32_t wakeCount   = 0;
RTC_DATA_ATTR uint32_t uptimeSec   = 0;
RTC_DATA_ATTR float    sensorHist[NN_LOOKBACK][NN_FEATURES];
RTC_DATA_ATTR int      histHead    = 0;
RTC_DATA_ATTR bool     histFull    = false;

// ====================================================================
//  CURRENT READINGS
// ====================================================================
float curAirTemp, curHum, curPres, curLux, curSoilPct, curSoilTemp;
float predMin = 0.0f, predMax = 0.0f;
float batPct  = 100.0f;
uint8_t alertLv = 0;

// ====================================================================
//  KALMAN FILTER
// ====================================================================
struct Kalman {
    float Q, R, x, P; bool init;
    void  setup(float q, float r) { Q=q; R=r; x=0; P=1; init=false; }
    float feed(float z) {
        if (!init) { x=z; P=1; init=true; return x; }
        P += Q;
        float K = P/(P+R);
        x += K*(z-x);
        P *= (1-K);
        return x;
    }
} kf[6];

// ====================================================================
//  NN INFERENCE
// ====================================================================
static float h1[NN_HIDDEN1], h2[NN_HIDDEN2], out[NN_OUTPUT_DIM];

static inline float pgmf(const float* a) { float v; memcpy_P(&v,a,4); return v; }
static inline float relu(float x) { return x > 0 ? x : 0; }

static void dense(const float* in, int ni, float* ou, int no,
                  const float* W, const float* B, bool act) {
    for (int j=0; j<no; j++) {
        float s = pgmf(&B[j]);
        for (int i=0; i<ni; i++) s += in[i]*pgmf(&W[i*no+j]);
        ou[j] = act ? relu(s) : s;
    }
}

void nnRun(float* iv) {
    dense(iv, NN_INPUT_DIM, h1, NN_HIDDEN1, W_HIDDEN1, B_HIDDEN1, true);
    dense(h1, NN_HIDDEN1,   h2, NN_HIDDEN2, W_HIDDEN2, B_HIDDEN2, true);
    dense(h2, NN_HIDDEN2,   out,NN_OUTPUT_DIM, W_OUTPUT, B_OUTPUT, false);
    predMin = constrain(out[0]*pgmf(&OUT_RANGE[0])+pgmf(&OUT_MIN[0]), -30.0f, 50.0f);
    predMax = constrain(out[1]*pgmf(&OUT_RANGE[1])+pgmf(&OUT_MIN[1]), -20.0f, 60.0f);
}

float normF(float raw, int i) {
    return (raw - pgmf(&FEAT_MIN[i])) / pgmf(&FEAT_RANGE[i]);
}

// ====================================================================
//  ALERT CLASSIFICATION
// ====================================================================
uint8_t classify() {
    bool night = (curLux < NIGHT_LUX);
    if (night && predMin < FROST_TEMP_C)               return 1;
    if (!night && predMax > HEAT_TEMP_C)               return 2;
    if (curSoilPct < DROUGHT_SOIL_PCT && predMax>30.0f) return 3;
    return 0;
}
const char* alertStr(uint8_t a) {
    switch(a) { case 1: return "FROST"; case 2: return "HEAT"; case 3: return "DROUGHT"; }
    return "SAFE";
}

// ====================================================================
//  SENSOR READING
// ====================================================================
float soilPct(int adc) {
    float p = 100.0f*(1.0f-(float)(adc-SOIL_WET_ADC)/(float)(SOIL_DRY_ADC-SOIL_WET_ADC));
    return constrain(p, 0.0f, 100.0f);
}
float batVolt() {
    float v = (analogRead(VBAT_ADC_PIN)/4095.0f)*3.3f*2.0f;
    return constrain((v-3.0f)/1.2f*100.0f, 0.0f, 100.0f);
}
bool readSensors() {
    float t = dht.readTemperature();
    float h = dht.readHumidity();
    if (isnan(t)||isnan(h)) return false;
    float p  = bmp.readSealevelPressure(ELEV_M);
    float lx = analogRead(LIGHT_ADC_PIN)*(120000.0f/4095.0f);
    float sm = soilPct(analogRead(SOIL_ADC_PIN));
    // ds18.requestTemperatures();
    // float st = ds18.getTempCByIndex(0);
    // if (st == DEVICE_DISCONNECTED_C) st = t - 2.0f;
    float st = t - 2.0f; // fallback if DS18B20 fails
    curAirTemp  = kf[0].feed(t);
    curHum      = kf[1].feed(h);
    curPres     = kf[2].feed(p);
    curLux      = kf[3].feed(lx);
    curSoilPct  = kf[4].feed(sm);
    curSoilTemp = kf[5].feed(st);
    batPct      = batVolt();
    return true;
}

// ====================================================================
//  OLED
// ====================================================================
void oledSplash() {
    oled.clearDisplay();
    oled.fillRect(0,0,128,11,SSD1306_WHITE);
    oled.setTextColor(SSD1306_BLACK); oled.setTextSize(1);
    oled.setCursor(10,2); oled.print("FARM SENTINEL");
    oled.setTextColor(SSD1306_WHITE);
    oled.setCursor(0,16); oled.printf("Zone%d: %.14s", ZONE_ID, ZONE_NAME);
    oled.setCursor(0,28); oled.printf("Wake #%lu", (unsigned long)wakeCount);
    oled.setCursor(0,40); oled.printf("Bat: %.0f%%", batPct);
    oled.display(); delay(2000);
}

void oledProgress(int s, int total) {
    oled.clearDisplay(); oled.setTextSize(1); oled.setTextColor(SSD1306_WHITE);
    oled.setCursor(0,0); oled.printf("Sample %d/%d", s, total);
    oled.drawRect(0,10,128,7,SSD1306_WHITE);
    int fill=(127)*s/total; if(fill>0) oled.fillRect(1,11,fill,5,SSD1306_WHITE);
    oled.setCursor(0,22); oled.printf("T:%.1fC H:%.0f%%", curAirTemp, curHum);
    oled.setCursor(0,32); oled.printf("Soil:%.0f%% ST:%.1fC", curSoilPct, curSoilTemp);
    oled.setCursor(0,42); oled.printf("P:%.0fhPa L:%.0f", curPres/100.0f, curLux);
    oled.display();
}

void oledResult() {
    oled.clearDisplay(); oled.setTextSize(1);
    bool alarm = alertLv > 0;
    oled.fillRect(0,0,128,11,alarm?SSD1306_WHITE:0);
    oled.setTextColor(alarm?SSD1306_BLACK:SSD1306_WHITE);
    oled.setCursor(2,2); oled.printf("%-10s  %s", ZONE_NAME, alertStr(alertLv));
    oled.setTextColor(SSD1306_WHITE);
    oled.setCursor(0,14); oled.printf("T:%.1fC  H:%.0f%%", curAirTemp, curHum);
    oled.setCursor(0,24); oled.printf("Soil:%.0f%%  ST:%.1fC", curSoilPct, curSoilTemp);
    oled.setCursor(0,34); oled.printf("P:%.0fhPa", curPres/100.0f);
    oled.drawLine(0,44,127,44,SSD1306_WHITE);
    oled.setCursor(0,47); oled.printf("6h> Mn:%.1f Mx:%.1f", predMin, predMax);
    oled.setCursor(0,57); oled.printf("Bat:%.0f%%  Lx:%.0f", batPct, curLux);
    oled.display();
}

// ====================================================================
//  ESP-NOW
// ====================================================================
uint8_t bcast[] = {0xFF,0xFF,0xFF,0xFF,0xFF,0xFF};

void onSent(const wifi_tx_info_t*, esp_now_send_status_t st) {
    Serial.println(st==ESP_NOW_SEND_SUCCESS?"[ESPNOW] OK":"[ESPNOW] FAIL");
}
void onRecv(const esp_now_recv_info_t*, const uint8_t* d, int len) {
    if (len != sizeof(FarmPacket)) return;
    FarmPacket p; memcpy(&p,d,sizeof(p));
    uint8_t z = p.zone_id % MAX_ZONES;
    zoneMap[z]=p; zoneActive[z]=true; zoneLastSeen[z]=millis();
    Serial.printf("[ESPNOW] Rx zone %d (%s) alert=%d\n",p.zone_id,p.zone_name,p.alert);
}

void espNowInit(bool hub) {
    WiFi.mode(WIFI_STA);
    if (esp_now_init()!=ESP_OK) { Serial.println("[ESPNOW] init fail"); return; }
    if (hub) {
        esp_now_register_recv_cb(onRecv);
    } else {
        esp_now_register_send_cb(onSent);
        esp_now_peer_info_t pi={};
        memcpy(pi.peer_addr,bcast,6); pi.encrypt=false;
        esp_now_add_peer(&pi);
    }
}

void sendPacket() {
    FarmPacket p;
    p.zone_id=ZONE_ID;
    strncpy(p.zone_name, ZONE_NAME, sizeof(p.zone_name)-1);
    p.zone_name[sizeof(p.zone_name)-1]='\0';
    p.air_temp=curAirTemp; p.humidity=curHum; p.pressure=curPres;
    p.lux=curLux; p.soil_pct=curSoilPct; p.soil_temp=curSoilTemp;
    p.pred_min=predMin; p.pred_max=predMax;
    p.alert=alertLv; p.battery_pct=batPct; p.uptime_s=uptimeSec;
    esp_now_send(bcast,(uint8_t*)&p,sizeof(p));
    // mirror into zone map
    zoneMap[ZONE_ID]=p; zoneActive[ZONE_ID]=true; zoneLastSeen[ZONE_ID]=millis();
}

// ====================================================================
//  HUB: FARM MAP JSON
// ====================================================================
void farmMapJson(String& out) {
    out="{\"zones\":["; bool first=true;
    for (int i=0;i<MAX_ZONES;i++) {
        if (!zoneActive[i]) continue;
        if (millis()-zoneLastSeen[i]>1800000UL) { zoneActive[i]=false; continue; }
        const FarmPacket& z=zoneMap[i];
        if (!first) out+=",";
        char buf[256];
        snprintf(buf,sizeof(buf),
            "{\"id\":%d,\"name\":\"%s\",\"air_temp\":%.1f,\"humidity\":%.1f,"
            "\"pressure\":%.0f,\"lux\":%.0f,\"soil_pct\":%.1f,\"soil_temp\":%.1f,"
            "\"pred_min\":%.1f,\"pred_max\":%.1f,\"alert\":%d,\"bat\":%.0f}",
            z.zone_id, z.zone_name, z.air_temp, z.humidity,
            z.pressure, z.lux, z.soil_pct, z.soil_temp,
            z.pred_min, z.pred_max, z.alert, z.battery_pct);
        out+=buf; first=false;
    }
    out+="]}";
}

// ====================================================================
//  HUB: BLE GATT
// ====================================================================
void runBLE() {
    BLEDevice::init("FarmSentinel-Hub");
    BLEServer*  srv = BLEDevice::createServer();
    BLEService* svc = srv->createService(SERVICE_UUID);

    String mapJson; farmMapJson(mapJson);
    BLECharacteristic* mapCh = svc->createCharacteristic(
        CHAR_FARM_MAP_UUID,
        BLECharacteristic::PROPERTY_READ|BLECharacteristic::PROPERTY_NOTIFY);
    mapCh->addDescriptor(new BLE2902());
    mapCh->setValue(mapJson.c_str());

    String alerts="";
    for (int i=0;i<MAX_ZONES;i++)
        if (zoneActive[i]) alerts+=String(i)+":"+String(zoneMap[i].alert)+";";
    BLECharacteristic* alCh = svc->createCharacteristic(
        CHAR_ALERTS_UUID,
        BLECharacteristic::PROPERTY_READ|BLECharacteristic::PROPERTY_NOTIFY);
    alCh->addDescriptor(new BLE2902()); alCh->setValue(alerts.c_str());

    char hub[64];
    int activeZones=0;
    for(int i=0;i<MAX_ZONES;i++) if(zoneActive[i]) activeZones++;
    snprintf(hub,sizeof(hub),"uptime:%lu,bat:%.0f,zones:%d",
             (unsigned long)uptimeSec, batPct, activeZones);
    BLECharacteristic* hubCh=svc->createCharacteristic(CHAR_HUB_UUID,BLECharacteristic::PROPERTY_READ);
    hubCh->setValue(hub);

    svc->start();
    BLEAdvertising* adv=BLEDevice::getAdvertising();
    adv->addServiceUUID(SERVICE_UUID); adv->setScanResponse(true);
    BLEDevice::startAdvertising();
    Serial.printf("[BLE] Advertising %d ms\n",BLE_WINDOW_MS);
    delay(BLE_WINDOW_MS);
    BLEDevice::stopAdvertising(); BLEDevice::deinit(true);
    Serial.println("[BLE] Done.");
}

// ====================================================================
//  HUB: HTTP
// ====================================================================
void setupHttp() {
    http.on("/api/farm", HTTP_GET, [](){
        String j; farmMapJson(j);
        http.sendHeader("Access-Control-Allow-Origin","*");
        http.send(200,"application/json",j);
    });
    http.on("/api/status", HTTP_GET, [](){
        char j[128];
        snprintf(j,sizeof(j),"{\"uptime\":%lu,\"bat\":%.1f,\"wake\":%lu}",
            (unsigned long)uptimeSec, batPct, (unsigned long)wakeCount);
        http.sendHeader("Access-Control-Allow-Origin","*");
        http.send(200,"application/json",String(j));
    });
    // POST a field sample (CSV line) for later fine-tune export
    http.on("/api/log", HTTP_POST, [](){
        String body=http.arg("plain");
        prefs.begin("log",false);
        uint32_t n=prefs.getUInt("n",0);
        char k[12]; snprintf(k,sizeof(k),"s%05u",n);
        prefs.putString(k,body);
        prefs.putUInt("n",n+1);
        prefs.end();
        http.send(200,"text/plain","OK");
    });
    // Download all logged field samples as CSV for Python fine-tuning
    http.on("/api/dump", HTTP_GET, [](){
        prefs.begin("log",true);
        uint32_t n=prefs.getUInt("n",0);
        String csv="air_temp,humidity,pressure,lux,soil_moisture,soil_temp\n";
        for(uint32_t i=0;i<n;i++){
            char k[12]; snprintf(k,sizeof(k),"s%05u",i);
            csv+=prefs.getString(k,"")+"\n";
        }
        prefs.end();
        http.sendHeader("Content-Disposition","attachment; filename=field_samples.csv");
        http.send(200,"text/csv",csv);
    });
    http.begin();
    Serial.printf("[HTTP] Ready at http://%s/\n", WiFi.localIP().toString().c_str());
}

// ====================================================================
//  LED
// ====================================================================
void blink(int n, int on=100, int off=100) {
    pinMode(STATUS_LED_PIN,OUTPUT);
    for(int i=0;i<n;i++){
        digitalWrite(STATUS_LED_PIN,HIGH); delay(on);
        digitalWrite(STATUS_LED_PIN,LOW);  delay(off);
    }
}

// ====================================================================
//  SETUP (runs each wake cycle)
// ====================================================================
void setup() {
    Serial.begin(115200); delay(300);
    wakeCount++;
    uptimeSec += (alertLv>0 ? SLEEP_ALERT_MIN : SLEEP_NORMAL_MIN)*60;

    Serial.printf("\n=== FarmSentinel Wake #%lu  Zone%d: %s  Hub:%s ===\n",
        (unsigned long)wakeCount, ZONE_ID, ZONE_NAME, IS_HUB?"YES":"NO");

    // ── Kalman setup ──────────────────────────────────────────────────
    kf[0].setup(0.02f,0.5f);   // air_temp
    kf[1].setup(0.02f,1.0f);   // humidity
    kf[2].setup(1.0f,10.0f);   // pressure
    kf[3].setup(20.0f,100.0f); // lux
    kf[4].setup(0.1f,1.0f);    // soil pct
    kf[5].setup(0.02f,0.5f);   // soil temp

    // ── Power on sensors ──────────────────────────────────────────────
    pinMode(SENSOR_PWR_PIN,OUTPUT);
    digitalWrite(SENSOR_PWR_PIN,HIGH);
    delay(2000);

    Wire.begin(21,22);
    dht.begin(); ds18.begin();
    analogReadResolution(12);

    bool bmpOk  = bmp.begin();
    bool oledOk = oled.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR);
    if (!bmpOk)  Serial.println("[WARN] BMP180 not found");
    if (!oledOk) Serial.println("[WARN] OLED not found");

    oled.clearDisplay(); oled.display();
    oledSplash();

    // ── Sample sensors ────────────────────────────────────────────────
    dht.readTemperature(); dht.readHumidity(); delay(2000); // discard first

    int good=0;
    for (int s=0; s<NUM_SAMPLES; s++) {
        bool ok = readSensors();
        if (ok) good++;
        Serial.printf("  [%d/%d] T:%.1f H:%.0f%% P:%.0f L:%.0f SM:%.0f%% ST:%.1f %s\n",
            s+1, NUM_SAMPLES, curAirTemp, curHum, curPres,
            curLux, curSoilPct, curSoilTemp, ok?"":"[FAIL]");
        oledProgress(s+1, NUM_SAMPLES);
        if (s < NUM_SAMPLES-1) delay(SAMPLE_INTERVAL_MS);
    }
    Serial.printf("  Valid samples: %d/%d\n", good, NUM_SAMPLES);

    // ── Push to history ring ──────────────────────────────────────────
    float fv[NN_FEATURES] = {
        normF(curAirTemp,0), normF(curHum,1), normF(curPres,2),
        normF(curLux,3),     normF(curSoilPct,4), normF(curSoilTemp,5)
    };
    memcpy(sensorHist[histHead % NN_LOOKBACK], fv, sizeof(fv));
    histHead++;
    if (histHead >= NN_LOOKBACK) histFull = true;

    // ── NN inference ──────────────────────────────────────────────────
    if (histFull || histHead >= NN_LOOKBACK) {
        float iv[NN_INPUT_DIM];
        for (int t=0; t<NN_LOOKBACK; t++) {
            int slot = (histHead - NN_LOOKBACK + t + NN_LOOKBACK*1000) % NN_LOOKBACK;
            memcpy(&iv[t*NN_FEATURES], sensorHist[slot], NN_FEATURES*sizeof(float));
        }
        nnRun(iv);
        Serial.printf("  NN: predMin=%.1f°C  predMax=%.1f°C\n", predMin, predMax);
    } else {
        // Rule-based fallback until 6 readings are in the ring
        predMin = curAirTemp - 4.0f;
        predMax = curAirTemp + 8.0f;
        Serial.printf("  Fallback predict: Min=%.1f Max=%.1f  (%d/%d readings)\n",
            predMin, predMax, histHead, NN_LOOKBACK);
    }

    alertLv = classify();
    Serial.printf("  Alert: %d (%s)\n", alertLv, alertStr(alertLv));
    oledResult();
    delay(3000);

    // ── ESP-NOW ───────────────────────────────────────────────────────
    espNowInit(IS_HUB);
    if (!IS_HUB) {
        sendPacket();
        delay(400);
        esp_now_deinit();
        WiFi.mode(WIFI_OFF);
    } else {
        // Hub listens 15s for incoming node packets
        Serial.println("[HUB] Collecting node packets (15s)...");
        delay(15000);
        sendPacket(); // include hub's own zone
    }

    // ── Hub services ──────────────────────────────────────────────────
    if (IS_HUB) {
        // BLE
        esp_now_deinit();
        WiFi.mode(WIFI_OFF);
        runBLE();

        // WiFi HTTP
        WiFi.mode(WIFI_STA);
        WiFi.begin(WIFI_SSID, WIFI_PASS);
        for (int t=0; t<20 && WiFi.status()!=WL_CONNECTED; t++) delay(500);
        if (WiFi.status()==WL_CONNECTED) {
            setupHttp();
            uint32_t until = millis() + (alertLv>0 ? HTTP_ALERT_MS : HTTP_IDLE_MS);
            while (millis()<until) { http.handleClient(); delay(10); }
            http.stop();
        } else {
            Serial.println("[WiFi] Failed — skipping HTTP.");
        }
        WiFi.disconnect(true); WiFi.mode(WIFI_OFF);
    }

    // ── LED feedback ──────────────────────────────────────────────────
    alertLv > 0 ? blink(alertLv*3, 200, 100) : blink(1, 600, 0);

    // ── Deep sleep ────────────────────────────────────────────────────
    int sleepMin = (alertLv>0) ? SLEEP_ALERT_MIN : SLEEP_NORMAL_MIN;
    Serial.printf("[SLEEP] %d min\n", sleepMin);
    Serial.flush();
    digitalWrite(SENSOR_PWR_PIN, LOW);
    esp_sleep_enable_timer_wakeup((uint64_t)sleepMin*60ULL*1000000ULL);
    esp_deep_sleep_start();
}

void loop() {}