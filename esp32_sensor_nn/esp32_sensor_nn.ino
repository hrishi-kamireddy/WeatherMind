#include "nn_weights.h"

#include <DHT.h>
#include <Wire.h>
#include <Adafruit_BMP085.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <Preferences.h>
#include <math.h>
#include <esp_now.h>
#include <WiFi.h>

struct WeatherResult {
    const char* name;
    const char* detail;
    int severity;
    const char* icon;
};

struct TelemetryRecord {
    float curTemp, curHum, curPres, curLux;
    float predTemp, predHum, predPres, predLux;
    float pressureTrend;
    int weatherSev;
    char weatherName[20];
    uint32_t wakeNum;
};

uint8_t broadcastAddress[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

void OnDataSent(const wifi_tx_info_t *info, esp_now_send_status_t status) {
    Serial.print("\r\nLast Packet Send Status:\t");
    Serial.println(status == ESP_NOW_SEND_SUCCESS ? "Delivery Success" : "Delivery Fail");
}

#define NN_HIDDEN1 16
#define NN_HIDDEN2 8
#define NN_FEATURES 4

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define SCREEN_ADDRESS 0x3C

#define SENSOR_POWER_PIN 15
#define DHT_PIN 4
#define LIGHT_PIN 34
#define CONFIRM_LED_PIN   14

#define SAMPLE_INTERVAL_MS 5000
#define NUM_SAMPLES 12
#define BLE_WINDOW_MS 30000

#define SLEEP_SEVERE 1
#define SLEEP_MILD 1
#define SLEEP_CLEAR 1

#define SERVICE_UUID             "e3a1f0b0-1234-5678-abcd-000000000001"
#define CHAR_CURRENT_UUID        "e3a1f0b0-1234-5678-abcd-000000000002"
#define CHAR_PREDICTED_UUID      "e3a1f0b0-1234-5678-abcd-000000000003"
#define CHAR_WEATHER_STATUS_UUID "e3a1f0b0-1234-5678-abcd-000000000004"
#define CHAR_ALERT_UUID          "e3a1f0b0-1234-5678-abcd-000000000005"
#define CHAR_TREND_UUID          "e3a1f0b0-1234-5678-abcd-000000000006"
#define CHAR_HISTORY_UUID        "e3a1f0b0-1234-5678-abcd-000000000007"

#define HISTORY_SLOTS 32
#define NVS_NAMESPACE "wmind"

struct KalmanFilter {
    float Q;
    float R;
    float x;
    float P;
    bool init;

    void reset(float q, float r) {
        Q = q; R = r; x = 0.0f; P = 1.0f; init = false;
    }

    float update(float z) {
        if (!init) { x = z; P = 1.0f; init = true; return x; }
        P = P + Q;
        float K = P / (P + R);
        x = x + K * (z - x);
        P = (1.0f - K) * P;
        return x;
    }
};

struct PressureTrend {
    static constexpr int WINDOW = 8;
    float samples[WINDOW];
    int count = 0;
    int head = 0;

    void push(float p) {
        samples[head % WINDOW] = p;
        head++;
        if (count < WINDOW) count++;
    }

    float slopePerMin() const {
        if (count < 2) return 0.0f;
        int n = count;
        float sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
        for (int i = 0; i < n; i++) {
            float x = (float)i;
            float y = samples[(head - n + i + WINDOW * 100) % WINDOW];
            sumX += x; sumY += y;
            sumXY += x * y; sumX2 += x * x;
        }
        float denom = n * sumX2 - sumX * sumX;
        if (fabsf(denom) < 1e-6f) return 0.0f;
        float slope = (n * sumXY - sumX * sumY) / denom;
        return slope * (60.0f / (SAMPLE_INTERVAL_MS / 1000.0f));
    }

    const char* label() const {
        float s = slopePerMin();
        if (s > 1.5f) return "RISING FAST";
        if (s > 0.4f) return "RISING";
        if (s < -1.5f) return "FALLING FAST";
        if (s < -0.4f) return "FALLING";
        return "STEADY";
    }
};



DHT dht(DHT_PIN, DHT11);
Adafruit_BMP085 bmp;
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
Preferences prefs;

KalmanFilter kf[4];
PressureTrend pressTrend;

RTC_DATA_ATTR int wakeCount = 0;
RTC_DATA_ATTR char prevWeatherName[20] = "UNKNOWN";
RTC_DATA_ATTR int prevSeverity = -1;
RTC_DATA_ATTR float prevPressureMb = 0.0f;
RTC_DATA_ATTR uint8_t historyHead = 0;

float sensorBuffer[NN_LOOKBACK][NN_FEATURES];
float inputVec[NN_INPUT_DIM];
float hidden1Buf[NN_HIDDEN1];
float hidden2Buf[NN_HIDDEN2];
float outputBuf[NN_OUTPUT_DIM];

float curTemp, curHum, curPres, curLux;
float predTemp = 0.0f, predHum = 0.0f, predPres = 0.0f, predLux = 0.0f;
float pressureSlopePaMin = 0.0f;

const char* weatherName = "UNKNOWN";
const char* weatherDetail = "";
const char* weatherIcon = "?";
int weatherSev = 0;
bool alertEscalated = false;

static inline float pgm_float(const float* addr) {
    float v; memcpy_P(&v, addr, sizeof(float)); return v;
}
static inline float relu(float x) { return x > 0.0f ? x : 0.0f; }
static inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

void denseForward(const float* in, int inN, float* out, int outN,
                  const float* W, const float* B, char act) {
    for (int j = 0; j < outN; j++) {
        float sum = pgm_float(&B[j]);
        for (int i = 0; i < inN; i++) {
            sum += in[i] * pgm_float(&W[i * outN + j]);
        }
        switch (act) {
            case 'r': sum = relu(sum); break;
            case 's': sum = sigmoidf(sum); break;
            default: break;
        }
        out[j] = sum;
    }
}

void nnPredict(float* in, float* out) {
    denseForward(in, NN_INPUT_DIM, hidden1Buf, NN_HIDDEN1, W_HIDDEN1, B_HIDDEN1, 'r');
    denseForward(hidden1Buf, NN_HIDDEN1, hidden2Buf, NN_HIDDEN2, W_HIDDEN2, B_HIDDEN2, 'r');
    denseForward(hidden2Buf, NN_HIDDEN2, out, NN_OUTPUT_DIM, W_OUTPUT, B_OUTPUT, 's');
}

float normalizeVal(float raw, int i) {
    return (raw - pgm_float(&FEAT_MIN[i])) / pgm_float(&FEAT_RANGE[i]);
}
float denormalizeVal(float n, int i) {
    return n * pgm_float(&FEAT_RANGE[i]) + pgm_float(&FEAT_MIN[i]);
}

float iqrMean(float* vals, int n) {
    float sorted[NUM_SAMPLES];
    memcpy(sorted, vals, n * sizeof(float));
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (sorted[j] < sorted[i]) { float tmp = sorted[i]; sorted[i] = sorted[j]; sorted[j] = tmp; }

    float q1 = sorted[n / 4];
    float q3 = sorted[(3 * n) / 4];
    float iqr = q3 - q1;
    float lo = q1 - 1.5f * iqr;
    float hi = q3 + 1.5f * iqr;

    float sum = 0.0f; int cnt = 0;
    for (int i = 0; i < n; i++)
        if (vals[i] >= lo && vals[i] <= hi) { sum += vals[i]; cnt++; }
    return cnt > 0 ? sum / cnt : 0.0f;
}

void applyPressureCorrection(float& predPres, float slopePaMin) {
    predPres += slopePaMin * 30.0f;
}

WeatherResult classifyWeather(float tempC, float humPct,
                              float pressPa, float lux,
                              float slopePaMin) {
    float pressMb = pressPa / 100.0f;
    bool rapidDrop = slopePaMin < -2.0f;

    if (tempC > 26.7f && pressMb < 980.0f && humPct >= 90.0f && lux < 50.0f)
        return {"HURRICANE", "Extreme low pressure, warm & humid", 3, "HUR"};
    if (tempC < -7.0f && pressMb < 995.0f && humPct > 80.0f && lux < 100.0f)
        return {"BLIZZARD", "Extreme cold + low pressure", 3, "BLZ"};
    if ((rapidDrop || pressMb < 1000.0f) && tempC > 18.0f && humPct > 70.0f)
        return {"THUNDERSTORM", "Low pressure, warm & humid — storm approach", 3, "THN"};

    if (tempC < 0.0f && pressMb < 1010.0f && humPct > 70.0f)
        return {"SNOW", "Sub-zero + humid + low pressure", 2, "SNW"};
    if (tempC >= 4.4f && tempC <= 26.7f && pressMb >= 990.0f && pressMb <= 1005.0f
        && humPct >= 60.0f && lux >= 500.0f && lux <= 2000.0f)
        return {"RAIN", "Low pressure, humid & overcast", 2, "RAN"};
    if (tempC > 38.0f && humPct < 40.0f && lux > 500.0f)
        return {"HEAT WAVE", "Extreme heat, dry & bright", 2, "HT!"};
    if (rapidDrop && pressMb < 1010.0f && lux < 100.0f)
        return {"STORM WATCH", "Rapid pressure drop detected", 2, "WRN"};

    if (humPct >= 95.0f && pressMb > 1013.0f && lux < 100.0f)
        return {"FOG", "Saturated humidity, low visibility", 1, "FOG"};
    if (pressMb < 1010.0f && humPct > 60.0f)
        return {"OVERCAST", "Low pressure, elevated humidity", 1, "OVC"};
    if (humPct >= 70.0f && humPct < 95.0f && pressMb >= 1008.0f)
        return {"CLOUDY", "Partly cloudy conditions", 1, "CLD"};

    return {"CLEAR", "No severe weather indicators", 0, "CLR"};
}

bool readSensors(float* temp, float* hum, float* pres, float* lux) {
    float t = dht.readTemperature();
    float h = dht.readHumidity();
    if (isnan(t) || isnan(h)) return false;
    if (t < -40.0f || t > 80.0f || h < 0.0f || h > 100.0f) return false;
    *temp = kf[0].update(t);
    *hum  = kf[1].update(h);
    *pres = kf[2].update((float)bmp.readSealevelPressure(515));
    float rawADC = (float)analogRead(LIGHT_PIN);
    *lux  = kf[3].update(rawADC * (632.0f / 4095.0f));
    return true;
}

void drawProgressBar(int x, int y, int w, int h, int pct) {
    display.drawRect(x, y, w, h, SSD1306_WHITE);
    int fill = (w - 2) * pct / 100;
    if (fill > 0) display.fillRect(x + 1, y + 1, fill, h - 2, SSD1306_WHITE);
}

void renderCollectingScreen(int sample, int total, float t, float h, float p, float l) {
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);

    display.setCursor(0, 0);
    display.print("WeatherMind");
    display.setCursor(80, 0);
    display.print("SAMPLING"); 

    int pct = sample * 100 / total;
    drawProgressBar(0, 11, 128, 6, pct);

    
    display.setCursor(0, 24);
    display.printf("T:%.1fC", t);
    display.setCursor(66, 24);
    display.printf("H:%.0f%%", h);

    display.setCursor(0, 34);
    display.printf("P:%.0fPa", p);
    display.setCursor(66, 34);
    display.printf("L:%.0f", l);

    display.drawLine(0, 50, 127, 50, SSD1306_WHITE); 
    
    display.setCursor(0, 55);
    display.printf("Progress: %d/%d", sample, total);

    display.display();
}

void drawDangerTriangle(int x, int y) {
    display.drawLine(x + 5, y,     x,      y + 9, SSD1306_WHITE);
    display.drawLine(x + 5, y,     x + 10, y + 9, SSD1306_WHITE);
    display.drawLine(x,     y + 9, x + 10, y + 9, SSD1306_WHITE);
    display.drawLine(x + 5, y + 3, x + 5, y + 6, SSD1306_WHITE);
    display.drawPixel(x + 5, y + 8, SSD1306_WHITE);
}

void renderResultScreen() {
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);

    // Current T + H on one row
    display.setCursor(0, 0);
    display.printf("T:%.1fC", curTemp);
    display.setCursor(66, 0);
    display.printf("H:%.0f%%", curHum);

    display.setCursor(0, 10);
    display.printf("P:%.0fPa", curPres);

    display.drawLine(0, 23, 127, 23, SSD1306_WHITE);
    display.fillRect(40, 20, 48, 7, SSD1306_BLACK);
    display.setCursor(43, 20);
    display.print("30 min");

    display.setCursor(0, 30);
    display.printf("T:%.1fC", predTemp);
    display.setCursor(66, 30);
    display.printf("H:%.0f%%", predHum);

    display.setCursor(0, 40);
    display.printf("P:%.0fPa", predPres);

    display.setCursor(0, 54);
    display.print(">> ");
    display.print(weatherName);

    if (weatherSev > 0) {
        drawDangerTriangle(115, 54);
    }

    display.display();
}

void saveToFlash(const TelemetryRecord& rec) {
    prefs.begin(NVS_NAMESPACE, false);
    char key[12];
    snprintf(key, sizeof(key), "rec%02u", (unsigned)(historyHead % HISTORY_SLOTS));
    prefs.putBytes(key, &rec, sizeof(TelemetryRecord));
    historyHead++;
    prefs.putUInt("head", historyHead);
    prefs.end();
}

int readHistory(TelemetryRecord* buf, int maxN) {
    prefs.begin(NVS_NAMESPACE, true);
    uint32_t head = prefs.getUInt("head", 0);
    int cnt = min((int)head, min(maxN, HISTORY_SLOTS));
    for (int i = 0; i < cnt; i++) {
        char key[12];
        uint32_t slot = (head - cnt + i) % HISTORY_SLOTS;
        snprintf(key, sizeof(key), "rec%02u", (unsigned)slot);
        prefs.getBytes(key, &buf[i], sizeof(TelemetryRecord));
    }
    prefs.end();
    return cnt;
}

void runBLE() {
    Serial.println("[BLE] Starting GATT server...");
    BLEDevice::init("WeatherMind");
    BLEServer* server = BLEDevice::createServer();
    BLEService* service = server->createService(SERVICE_UUID);

    auto addChar = [&](const char* uuid, const char* value) -> BLECharacteristic* {
        BLECharacteristic* ch = service->createCharacteristic(
            uuid, BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY);
        ch->addDescriptor(new BLE2902());
        ch->setValue(value);
        return ch;
    };

    char curStr[64];
    snprintf(curStr, sizeof(curStr), "%.1f,%.0f,%.0f,%.0f",
             curTemp, curHum, curPres, curLux);
    addChar(CHAR_CURRENT_UUID, curStr);

    char predStr[64];
    snprintf(predStr, sizeof(predStr), "%.1f,%.0f,%.0f,%.0f",
             predTemp, predHum, predPres, curLux);
    addChar(CHAR_PREDICTED_UUID, predStr);

    char wStr[96];
    snprintf(wStr, sizeof(wStr), "%s|%d|%s", weatherName, weatherSev, weatherDetail);
    addChar(CHAR_WEATHER_STATUS_UUID, wStr);

    char alertStr[4];
    snprintf(alertStr, sizeof(alertStr), "%d", alertEscalated ? 1 : 0);
    addChar(CHAR_ALERT_UUID, alertStr);

    char trendStr[32];
    snprintf(trendStr, sizeof(trendStr), "%s|%.2f", pressTrend.label(), pressureSlopePaMin);
    addChar(CHAR_TREND_UUID, trendStr);

    TelemetryRecord hist[8];
    int hcnt = readHistory(hist, 8);
    String histStr = "";
    for (int i = 0; i < hcnt; i++) {
        char line[64];
        snprintf(line, sizeof(line), "%.1f,%.0f,%.1f,%s;",
                 hist[i].curTemp, hist[i].curHum,
                 hist[i].curPres / 100.0f, hist[i].weatherName);
        histStr += line;
    }
    BLECharacteristic* histCh = service->createCharacteristic(
        CHAR_HISTORY_UUID, BLECharacteristic::PROPERTY_READ);
    histCh->setValue(histStr.c_str());

    service->start();

    BLEAdvertising* adv = BLEDevice::getAdvertising();
    adv->addServiceUUID(SERVICE_UUID);
    adv->setScanResponse(true);
    adv->setMinPreferred(0x06);
    BLEDevice::startAdvertising();

    Serial.printf("[BLE] Advertising for %d ms...\n", BLE_WINDOW_MS);
    delay(BLE_WINDOW_MS);

    BLEDevice::stopAdvertising();
    BLEDevice::deinit(true);
    Serial.println("[BLE] Done.");
}

void logSection(const char* title) {
    Serial.printf("\n┌─────────────────────────────────────────\n│ %s\n└─────────────────────────────────────────\n", title);
}

void sendEspNow(const TelemetryRecord& data) {
    pinMode(CONFIRM_LED_PIN, OUTPUT);
    for(int i = 0; i < 5; i++) {
        digitalWrite(CONFIRM_LED_PIN, HIGH);  
        delay(150);
        digitalWrite(CONFIRM_LED_PIN, LOW);
        delay(150);
    }
    WiFi.mode(WIFI_STA);
    if (esp_now_init() != ESP_OK) {
        return;
    }

    esp_now_register_send_cb(OnDataSent);

    esp_now_peer_info_t peerInfo = {};
    memcpy(peerInfo.peer_addr, broadcastAddress, 6);
    peerInfo.encrypt = false;

    if (esp_now_add_peer(&peerInfo) == ESP_OK) {
        esp_now_send(broadcastAddress, (uint8_t*)&data, sizeof(data));
    }

    delay(200);
    digitalWrite(CONFIRM_LED_PIN, LOW);

    esp_now_deinit();
    WiFi.mode(WIFI_OFF);
}

void renderTransmissionScreen(bool sent) {
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);

    display.setCursor(0, 0);
    display.print("WeatherMind");

    display.drawLine(0, 12, 127, 12, SSD1306_WHITE);

    if (sent) {
        display.setCursor(0, 25);
        display.print("Weather: ALERT");

        display.setCursor(0, 40);
        display.print("Sending to modules...");
    } else {
        display.setCursor(0, 25);
        display.print("Weather: CLEAR");

        display.setCursor(0, 40);
        display.print("Needs no transmission");
    }

    display.display();
}

void setup() {
    Serial.begin(115200);
    wakeCount++;

    logSection("WeatherMind Pro — Wake");
    Serial.printf("  Wake #%d\n", wakeCount);

    kf[0].reset(0.01f, 0.5f);
    kf[1].reset(0.01f, 1.0f);
    kf[2].reset(0.5f, 5.0f);
    kf[3].reset(5.0f, 20.0f);

    pinMode(SENSOR_POWER_PIN, OUTPUT);
    digitalWrite(SENSOR_POWER_PIN, HIGH);
    delay(3000);

    Wire.begin(21, 22);
    dht.begin();
    analogReadResolution(12);
    pinMode(LIGHT_PIN, INPUT);

    if (!bmp.begin()) {
        Serial.println("[ERROR] BMP180 not found — halting");
        if (!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
            while (true) delay(1000);
        }
        display.clearDisplay();
        display.setTextSize(1);
        display.setTextColor(SSD1306_WHITE);
        display.setCursor(0, 20);
        display.println("BMP180 ERROR");
        display.display();
        while (true) delay(1000);
    }

    if (!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
        Serial.println("[ERROR] OLED not found — halting");
        while (true) delay(1000);
    }

    display.clearDisplay();
    display.display();
    delay(200);

    display.fillRect(0, 0, 128, 11, SSD1306_WHITE);
    display.setTextColor(SSD1306_BLACK);
    display.setTextSize(1);
    display.setCursor(32, 2); 
    display.print("WEATHERMIND"); 

    display.setTextColor(SSD1306_WHITE);
    
    display.setCursor(10, 28); 
    display.print("by Shadow Mechanics");
    
    display.drawLine(20, 39, 108, 39, SSD1306_WHITE);

    display.setCursor(42, 45);
    display.printf("Session #%d", wakeCount);

    display.display();
    delay(2500);

    dht.readTemperature();
    dht.readHumidity();
    delay(2500);

    logSection("Data Collection");

    float rawT[NUM_SAMPLES], rawH[NUM_SAMPLES], rawP[NUM_SAMPLES], rawL[NUM_SAMPLES];
    int samplesOk = 0;

    for (int s = 0; s < NUM_SAMPLES; s++) {
        float t, h, p, l;
        bool ok = readSensors(&t, &h, &p, &l);

        if (ok) {
            sensorBuffer[s][0] = normalizeVal(t, 0);
            sensorBuffer[s][1] = normalizeVal(h, 1);
            sensorBuffer[s][2] = normalizeVal(p, 2);
            sensorBuffer[s][3] = normalizeVal(l, 3);

            rawT[s] = t; rawH[s] = h; rawP[s] = p; rawL[s] = l;
            pressTrend.push(p);
            curTemp = t; curHum = h; curPres = p; curLux = l;
            samplesOk++;
            Serial.printf("  [%2d/%d] T:%5.1fC H:%4.1f%% P:%7.0fPa L:%5.0f  [KF]\n",
                          s + 1, NUM_SAMPLES, t, h, p, l);
        } else {
            if (s > 0) {
                for (int f = 0; f < NN_FEATURES; f++)
                    sensorBuffer[s][f] = sensorBuffer[s - 1][f];
                rawT[s] = curTemp; rawH[s] = curHum;
                rawP[s] = curPres; rawL[s] = curLux;
            }
            Serial.printf("  [%2d/%d] DHT read failed — using KF estimate\n", s + 1, NUM_SAMPLES);
        }

        renderCollectingScreen(s + 1, NUM_SAMPLES, curTemp, curHum, curPres, curLux);

        if (s < NUM_SAMPLES - 1) delay(SAMPLE_INTERVAL_MS);
    }

    Serial.printf("  Collected %d/%d valid samples\n", samplesOk, NUM_SAMPLES);

    if (samplesOk > 2) {
        curTemp = iqrMean(rawT, samplesOk);
        curHum = iqrMean(rawH, samplesOk);
        curPres = iqrMean(rawP, samplesOk);
        curLux = iqrMean(rawL, samplesOk);
    }

    pressureSlopePaMin = pressTrend.slopePerMin();
    Serial.printf("  Pressure trend: %.2f Pa/min  [%s]\n",
                  pressureSlopePaMin, pressTrend.label());

    logSection("Neural Network Inference");

    if (samplesOk >= NUM_SAMPLES / 2) {
        for (int s = 0; s < NN_LOOKBACK; s++) {
            for (int f = 0; f < NN_FEATURES; f++) {
                inputVec[s * NN_FEATURES + f] = sensorBuffer[s][f];
            }
        }

        nnPredict(inputVec, outputBuf);

        Serial.printf("Raw Norm Temp: %.6f\n", outputBuf[0]);
        Serial.printf("Raw Norm Hum:  %.6f\n", outputBuf[1]);
        Serial.printf("Raw Norm Pres: %.6f\n", outputBuf[2]);

        predTemp = denormalizeVal(outputBuf[0], 0);
        predHum  = denormalizeVal(outputBuf[1], 1);
        predPres = denormalizeVal(outputBuf[2], 2);
        predLux = curLux;

        applyPressureCorrection(predPres, pressureSlopePaMin);

        predTemp = constrain(predTemp, -50.0f, 80.0f);
        predHum = constrain(predHum, 0.0f, 100.0f);
        predPres = constrain(predPres, 85000.0f, 108000.0f);
        predLux = constrain(predLux, 0.0f, 632.0f);

        WeatherResult w = classifyWeather(predTemp, predHum, predPres, curLux, pressureSlopePaMin);
        weatherName = w.name;
        weatherDetail = w.detail;
        weatherSev = w.severity;
        weatherIcon = w.icon;

        alertEscalated = (weatherSev > prevSeverity && prevSeverity >= 0);
        strncpy(prevWeatherName, weatherName, sizeof(prevWeatherName) - 1);
        prevWeatherName[sizeof(prevWeatherName) - 1] = '\0';
        prevSeverity = weatherSev;
        prevPressureMb = curPres / 100.0f;

        static const char* sevLabel[] = {"CLEAR", "MILD", "MODERATE", "SEVERE"};
        Serial.printf("  Pred   T:%.1fC  H:%.1f%%  P:%.0fPa  L:%.1f\n",
                      predTemp, predHum, predPres, predLux);
        Serial.printf("  Result %s  [%s]  Escalated:%s\n",
                      weatherName, sevLabel[min(weatherSev,3)],
                      alertEscalated ? "YES" : "no");
    } else {
        weatherName = "NO DATA";
        weatherDetail = "Insufficient samples";
        weatherSev = 0;
        Serial.println("  Skipped — too few valid samples");
    }

    TelemetryRecord rec;
    rec.curTemp = curTemp;  rec.curHum = curHum;
    rec.curPres = curPres;   rec.curLux = curLux;
    rec.predTemp = predTemp; rec.predHum = predHum;
    rec.predPres = predPres; rec.predLux = predLux;
    rec.pressureTrend = pressureSlopePaMin;
    rec.weatherSev = weatherSev;
    rec.wakeNum = wakeCount;
    strncpy(rec.weatherName, weatherName, sizeof(rec.weatherName) - 1);
    rec.weatherName[sizeof(rec.weatherName) - 1] = '\0';
    saveToFlash(rec);
    Serial.printf("  Saved record #%u to NVS flash\n", wakeCount);

    logSection("ESP-NOW Transmission");
    bool sent = false;

    if (weatherSev > 0) {
        sendEspNow(rec);
        sent = true;

    } else {
        pinMode(CONFIRM_LED_PIN, OUTPUT);
        for(int i = 0; i < 3; i++) {
            digitalWrite(CONFIRM_LED_PIN, HIGH);
            delay(150);
            digitalWrite(CONFIRM_LED_PIN, LOW);
            delay(150);
        }
        Serial.println("Skipped — weather is CLEAR, no transmission needed");
        sent = false;
    }

    renderTransmissionScreen(sent);

    delay(6000);

    renderResultScreen();

    digitalWrite(SENSOR_POWER_PIN, LOW);

    logSection("BLE Broadcast");
    runBLE();
    
    int sleepMin;
    switch (weatherSev) {
        case 3: sleepMin = SLEEP_SEVERE; break;
        case 2: sleepMin = SLEEP_MILD; break;
        default: sleepMin = SLEEP_CLEAR; break;
    }

    logSection("Deep Sleep");
    Serial.printf("  Weather severity: %d  ->  sleeping %d min\n", weatherSev, sleepMin);
    Serial.printf("  OLED will retain last display during sleep.\n");
    Serial.flush();

    esp_sleep_enable_timer_wakeup((uint64_t)sleepMin * 60ULL * 1000000ULL);
    esp_deep_sleep_start();
}

void loop() {
}