/*
 * ═══════════════════════════════════════════════════════════════
 * WeatherMind — ESP32 NN + BLE + ESP-NOW (Light Sleep)
 * ═══════════════════════════════════════════════════════════════
 *
 * Light sleep: BLE stays alive, phone connects anytime.
 * ESP-NOW: broadcasts weather to nearby ESP32s.
 *
 * Each cycle:
 *   1. Wake from light sleep
 *   2. Power on sensors, collect 12 readings (~1 min)
 *   3. NN inference -> classify weather
 *   4. Update OLED + notify BLE + broadcast ESP-NOW
 *   5. Power off sensors, light sleep
 *
 * SLEEP TIME:
 *   Currently 1 minute for testing.
 *   For production, change SLEEP_MINUTES from 1 to 29.
 *
 * ESP-NOW SETUP:
 *   Change peerAddress to the other ESP32's MAC address.
 *   Use 0xFF x6 to broadcast to all nearby ESP32s.
 *   Each board needs a unique DEVICE_NAME.
 *
 * NN: 48 -> 16 (ReLU) -> 8 (ReLU) -> 4 (Sigmoid)
 * ═══════════════════════════════════════════════════════════════
 */

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
#include <esp_now.h>
#include <WiFi.h>

// ─── Device Identity ─────────────────────────────────────────
// Change this on each board so you know who sent what
#define DEVICE_NAME "WM-Station1"

// ─── Display ─────────────────────────────────────────────────
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define SCREEN_ADDRESS 0x3C

// ─── Pins ────────────────────────────────────────────────────
#define SENSOR_POWER_PIN 15
#define DHT_PIN          4
#define LIGHT_PIN        2

// ─── Timing ──────────────────────────────────────────────────
#define SAMPLE_INTERVAL_MS 5000
#define NUM_SAMPLES        12

// ┌─────────────────────────────────────────────────────────┐
// │  SLEEP TIME: Change this to 29 for production           │
// └─────────────────────────────────────────────────────────┘
#define SLEEP_MINUTES 1

// ─── BLE UUIDs ───────────────────────────────────────────────
#define SERVICE_UUID              "e3a1f0b0-1234-5678-abcd-000000000001"
#define CHAR_CURRENT_UUID         "e3a1f0b0-1234-5678-abcd-000000000002"
#define CHAR_PREDICTED_UUID       "e3a1f0b0-1234-5678-abcd-000000000003"
#define CHAR_WEATHER_STATUS_UUID  "e3a1f0b0-1234-5678-abcd-000000000004"
#define CHAR_ALERT_UUID           "e3a1f0b0-1234-5678-abcd-000000000005"

// ─── ESP-NOW Peer ────────────────────────────────────────────
// Set to the OTHER ESP32's MAC address.
// Use FF:FF:FF:FF:FF:FF to broadcast to ALL nearby ESP32s.
// Find your MAC in Serial Monitor at boot.
uint8_t peerAddress[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

// ─── ESP-NOW Message Structure ───────────────────────────────
#define MSG_TYPE_WEATHER 1
#define MSG_TYPE_TEXT    2

typedef struct {
    uint8_t  msgType;          // 1 = weather, 2 = text
    char     senderName[16];   // device name
    char     weatherName[20];  // classification
    int      severity;         // 0-3
    float    temperature;      // predicted temp
    float    humidity;         // predicted humidity
    float    pressure;         // predicted pressure
    float    lux;              // predicted lux
    char     textMsg[64];      // custom text message
} EspNowMessage;

EspNowMessage outgoingMsg;
EspNowMessage incomingMsg;
volatile bool newMessageReceived = false;

// ─── Sensor objects ──────────────────────────────────────────
DHT dht(DHT_PIN, DHT11);
Adafruit_BMP085 bmp;
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// ─── BLE objects ─────────────────────────────────────────────
BLEServer*         pServer        = nullptr;
BLECharacteristic* pCurrentChar   = nullptr;
BLECharacteristic* pPredictedChar = nullptr;
BLECharacteristic* pStatusChar    = nullptr;
BLECharacteristic* pAlertChar     = nullptr;
bool deviceConnected    = false;
bool oldDeviceConnected = false;

// ─── NN buffers ──────────────────────────────────────────────
float sensorBuffer[NN_LOOKBACK][4];
float inputVec[NN_INPUT_DIM];
float hidden1Buf[NN_HIDDEN1];
float hidden2Buf[NN_HIDDEN2];
float outputBuf[NN_OUTPUT_DIM];

// ─── State ───────────────────────────────────────────────────
float curTemp = 0, curHum = 0, curPres = 0, curLux = 0;
float predTemp = 0, predHum = 0, predPres = 0, predLux = 0;
const char* weatherName   = "WAITING";
const char* weatherDetail = "First reading pending";
int         weatherSev    = 0;
bool        hasPrediction = false;
int         cycleCount    = 0;
String      prevAlertType = "UNKNOWN";

// ═══════════════════════════════════════════════════════════════
//  Weather Classification
// ═══════════════════════════════════════════════════════════════
struct WeatherResult {
    const char* name;
    const char* detail;
    int         severity;
};

WeatherResult classifyWeather(float tempC, float humPct, float pressPa, float lux) {
    float pressMb = pressPa / 100.0;

    if (tempC > 26.7 && pressMb < 980.0 && humPct >= 90.0 && lux < 50.0)
        return {"HURRICANE", "Extreme low pressure + warm + humid", 3};

    if (tempC < -7.0 && pressMb < 995.0 && humPct > 80.0 && lux < 100.0)
        return {"BLIZZARD", "Extreme cold + low pressure", 3};

    if (tempC > 18.0 && pressMb < 1000.0 && humPct > 70.0 && lux >= 100.0 && lux <= 1000.0)
        return {"THUNDERSTORM", "Low pressure + warm + humid", 3};

    if (tempC < 0.0 && pressMb < 1010.0 && humPct > 70.0 && lux >= 1000.0 && lux <= 5000.0)
        return {"SNOW", "Below freezing + humid", 2};

    if (tempC >= 4.4 && tempC <= 26.7 && pressMb >= 990.0 && pressMb <= 1005.0 &&
        humPct >= 60.0 && lux >= 500.0 && lux <= 2000.0)
        return {"RAIN", "Low pressure + humid + overcast", 2};

    if (humPct >= 95.0 && pressMb > 1013.0 && lux < 100.0)
        return {"FOG", "Saturated humidity + low visibility", 1};

    if (tempC > 38.0 && humPct < 40.0 && lux > 500.0)
        return {"HEAT WAVE", "Extreme heat + dry + bright", 2};

    return {"CLEAR", "No severe weather detected", 0};
}

// ═══════════════════════════════════════════════════════════════
//  ESP-NOW Callbacks
// ═══════════════════════════════════════════════════════════════
void onDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
    Serial.printf("ESP-NOW send: %s\n",
                  status == ESP_NOW_SEND_SUCCESS ? "OK" : "FAIL");
}

void onDataRecv(const esp_now_recv_info *info, const uint8_t *data, int len) {
    if (len > sizeof(incomingMsg)) len = sizeof(incomingMsg);
    memcpy(&incomingMsg, data, len);
    newMessageReceived = true;
}

// ═══════════════════════════════════════════════════════════════
//  ESP-NOW Send Functions
// ═══════════════════════════════════════════════════════════════
void sendWeatherESPNow() {
    memset(&outgoingMsg, 0, sizeof(outgoingMsg));
    outgoingMsg.msgType = MSG_TYPE_WEATHER;
    strncpy(outgoingMsg.senderName, DEVICE_NAME, 15);
    strncpy(outgoingMsg.weatherName, weatherName, 19);
    outgoingMsg.severity    = weatherSev;
    outgoingMsg.temperature = predTemp;
    outgoingMsg.humidity    = predHum;
    outgoingMsg.pressure    = predPres;
    outgoingMsg.lux         = predLux;

    esp_now_send(peerAddress, (uint8_t *)&outgoingMsg, sizeof(outgoingMsg));
}

void sendTextESPNow(const char* message) {
    memset(&outgoingMsg, 0, sizeof(outgoingMsg));
    outgoingMsg.msgType = MSG_TYPE_TEXT;
    strncpy(outgoingMsg.senderName, DEVICE_NAME, 15);
    strncpy(outgoingMsg.textMsg, message, 63);

    esp_now_send(peerAddress, (uint8_t *)&outgoingMsg, sizeof(outgoingMsg));
}

// ═══════════════════════════════════════════════════════════════
//  BLE Callbacks
// ═══════════════════════════════════════════════════════════════
class ServerCallbacks : public BLEServerCallbacks {
    void onConnect(BLEServer* s) override {
        deviceConnected = true;
        Serial.println("BLE: Device connected");
    }
    void onDisconnect(BLEServer* s) override {
        deviceConnected = false;
        Serial.println("BLE: Device disconnected");
    }
};

// ═══════════════════════════════════════════════════════════════
//  NN Math
// ═══════════════════════════════════════════════════════════════
static inline float pgm_float(const float* addr) {
    float v; memcpy_P(&v, addr, sizeof(float)); return v;
}
static inline float relu(float x)     { return x > 0.0f ? x : 0.0f; }
static inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

void denseForward(const float* in, int inN, float* out, int outN,
                  const float* W, const float* B, char act) {
    for (int j = 0; j < outN; j++) {
        float sum = pgm_float(&B[j]);
        for (int i = 0; i < inN; i++)
            sum += in[i] * pgm_float(&W[i * outN + j]);
        if (act == 'r')      sum = relu(sum);
        else if (act == 's') sum = sigmoidf(sum);
        out[j] = sum;
    }
}

void nnPredict(float* in, float* out) {
    denseForward(in,         NN_INPUT_DIM, hidden1Buf, NN_HIDDEN1, W_HIDDEN1, B_HIDDEN1, 'r');
    denseForward(hidden1Buf, NN_HIDDEN1,   hidden2Buf, NN_HIDDEN2, W_HIDDEN2, B_HIDDEN2, 'r');
    denseForward(hidden2Buf, NN_HIDDEN2,   out,        NN_OUTPUT_DIM, W_OUTPUT, B_OUTPUT, 's');
}

float normalizeVal(float raw, int i) {
    return (raw - pgm_float(&FEAT_MIN[i])) / pgm_float(&FEAT_RANGE[i]);
}
float denormalizeVal(float n, int i) {
    return n * pgm_float(&FEAT_RANGE[i]) + pgm_float(&FEAT_MIN[i]);
}

// ═══════════════════════════════════════════════════════════════
//  Read Sensors
// ═══════════════════════════════════════════════════════════════
bool readSensors(float* temp, float* hum, float* pres, float* lux) {
    *temp = dht.readTemperature();
    *hum  = dht.readHumidity();
    if (isnan(*temp) || isnan(*hum)) return false;
    *pres = bmp.readSealevelPressure(515);
    *lux  = (float)analogRead(LIGHT_PIN);
    return true;
}

// ═══════════════════════════════════════════════════════════════
//  Update OLED
// ═══════════════════════════════════════════════════════════════
void updateOLED() {
    display.clearDisplay();
    display.setTextColor(SSD1306_WHITE);

    display.setTextSize(1);
    display.setCursor(0, 0);
    display.print("NOW");

    display.setCursor(0, 10);
    display.print("T:");
    display.print(curTemp, 1);
    display.print("C  H:");
    display.print(curHum, 0);
    display.print("%");

    display.setCursor(0, 20);
    display.print("P:");
    display.print(curPres, 0);
    display.print(" L:");
    display.print((int)curLux);

    display.drawLine(0, 30, 127, 30, SSD1306_WHITE);

    display.setTextSize(1);
    display.setCursor(0, 34);
    display.print("30m: ");
    display.print(predTemp, 1);
    display.print("C");

    display.setTextSize(2);
    display.setCursor(0, 46);
    display.print(weatherName);

    display.display();
}

// ═══════════════════════════════════════════════════════════════
//  Show incoming ESP-NOW message on OLED
// ═══════════════════════════════════════════════════════════════
void showIncomingMessage() {
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);

    if (incomingMsg.msgType == MSG_TYPE_WEATHER) {
        display.println("== INCOMING ==");
        display.printf("From: %s\n", incomingMsg.senderName);
        display.println();
        display.setTextSize(2);
        display.println(incomingMsg.weatherName);
        display.setTextSize(1);
        display.printf("Sev:%d T:%.1fC\n", incomingMsg.severity, incomingMsg.temperature);
    } else {
        display.println("== MESSAGE ==");
        display.printf("From: %s\n", incomingMsg.senderName);
        display.println();
        display.println(incomingMsg.textMsg);
    }

    display.display();
    delay(5000);
    updateOLED();
}

// ═══════════════════════════════════════════════════════════════
//  Update BLE
// ═══════════════════════════════════════════════════════════════
void updateBLE() {
    char buf[80];

    snprintf(buf, sizeof(buf), "%.1f,%.1f,%.0f,%.0f", curTemp, curHum, curPres, curLux);
    pCurrentChar->setValue(buf);
    if (deviceConnected) pCurrentChar->notify();

    if (!hasPrediction) return;

    snprintf(buf, sizeof(buf), "%.1f,%.1f,%.0f,%.1f", predTemp, predHum, predPres, predLux);
    pPredictedChar->setValue(buf);
    if (deviceConnected) pPredictedChar->notify();

    char statusBuf[120];
    snprintf(statusBuf, sizeof(statusBuf), "%s|%d|%s", weatherName, weatherSev, weatherDetail);
    pStatusChar->setValue(statusBuf);
    if (deviceConnected) pStatusChar->notify();

    if (String(weatherName) != prevAlertType) {
        char alertBuf[120];
        if (weatherSev >= 2)
            snprintf(alertBuf, sizeof(alertBuf), "ALERT:%s|%s", weatherName, weatherDetail);
        else
            snprintf(alertBuf, sizeof(alertBuf), "OK:%s|%s", weatherName, weatherDetail);
        pAlertChar->setValue(alertBuf);
        if (deviceConnected) pAlertChar->notify();
        Serial.printf(">>> BLE ALERT: %s\n", alertBuf);
        prevAlertType = String(weatherName);
    }
}

// ═══════════════════════════════════════════════════════════════
//  Run Full Cycle: collect -> predict -> display -> broadcast
// ═══════════════════════════════════════════════════════════════
void runCycle() {
    cycleCount++;
    Serial.printf("\n=========================================\n");
    Serial.printf("  WeatherMind - cycle #%d\n", cycleCount);
    Serial.printf("=========================================\n");

    // ── Power on sensors ─────────────────────────────────────
    digitalWrite(SENSOR_POWER_PIN, HIGH);
    delay(3000);

    // ── Re-init sensors ──────────────────────────────────────
    Wire.begin(21, 22);
    dht.begin();

    if (!bmp.begin()) {
        Serial.println("BMP180 not found!");
        return;
    }

    // ── Show collecting ──────────────────────────────────────
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.println("Collecting data...");
    display.println("12 readings / 1 min");
    display.display();

    // ── Flush DHT11 ──────────────────────────────────────────
    dht.readTemperature();
    dht.readHumidity();
    delay(2500);

    // ── Collect 12 readings ──────────────────────────────────
    int samplesCollected = 0;

    for (int s = 0; s < NUM_SAMPLES; s++) {
        float t, h, p, l;

        if (readSensors(&t, &h, &p, &l)) {
            sensorBuffer[s][0] = normalizeVal(t, 0);
            sensorBuffer[s][1] = normalizeVal(h, 1);
            sensorBuffer[s][2] = normalizeVal(p, 2);
            sensorBuffer[s][3] = normalizeVal(l, 3);

            curTemp = t; curHum = h; curPres = p; curLux = l;
            samplesCollected++;

            Serial.printf("  [%2d/%d] T:%.1fC  H:%.1f%%  P:%.0fPa  L:%.0f\n",
                          s + 1, NUM_SAMPLES, t, h, p, l);
        } else {
            Serial.printf("  [%2d/%d] DHT failed, copying previous\n", s + 1, NUM_SAMPLES);
            if (s > 0)
                for (int f = 0; f < 4; f++)
                    sensorBuffer[s][f] = sensorBuffer[s - 1][f];
        }

        // Progress on OLED
        display.clearDisplay();
        display.setCursor(0, 0);
        display.printf("Collecting %d/%d\n", s + 1, NUM_SAMPLES);
        if (samplesCollected > 0) {
            display.printf("T:%.1fC H:%.1f%%\n", curTemp, curHum);
            display.printf("P:%.0fPa L:%.0f\n", curPres, curLux);
        }
        display.display();

        // Check for incoming ESP-NOW during collection
        if (newMessageReceived) {
            newMessageReceived = false;
            Serial.printf("ESP-NOW received from %s during collection\n",
                          incomingMsg.senderName);
        }

        if (s < NUM_SAMPLES - 1)
            delay(SAMPLE_INTERVAL_MS);
    }

    // ── NN inference ─────────────────────────────────────────
    if (samplesCollected >= NUM_SAMPLES / 2) {
        for (int s = 0; s < NN_LOOKBACK; s++)
            for (int f = 0; f < 4; f++)
                inputVec[s * 4 + f] = sensorBuffer[s][f];

        nnPredict(inputVec, outputBuf);

        predTemp = denormalizeVal(outputBuf[0], 0);
        predHum  = denormalizeVal(outputBuf[1], 1);
        predPres = denormalizeVal(outputBuf[2], 2);
        predLux  = denormalizeVal(outputBuf[3], 3);

        WeatherResult w = classifyWeather(predTemp, predHum, predPres, predLux);
        weatherName   = w.name;
        weatherDetail = w.detail;
        weatherSev    = w.severity;
        hasPrediction = true;

        const char* sevLabel[] = {"OK", "MILD", "MODERATE", "SEVERE"};
        Serial.println("────────────────────────────────────────");
        Serial.printf("  PRED -> T:%.1fC  H:%.1f%%  P:%.0fPa  L:%.1f\n",
                      predTemp, predHum, predPres, predLux);
        Serial.printf("  WEATHER -> %s [%s]: %s\n",
                      weatherName, sevLabel[weatherSev], weatherDetail);
        Serial.println("────────────────────────────────────────");
    } else {
        weatherName = "NO DATA";
        weatherDetail = "Sensor errors";
        weatherSev = 0;
    }

    // ── Power off sensors ────────────────────────────────────
    digitalWrite(SENSOR_POWER_PIN, LOW);

    // ── Update everything ────────────────────────────────────
    updateOLED();
    updateBLE();

    // ── Broadcast to other ESP32s ────────────────────────────
    if (hasPrediction) {
        sendWeatherESPNow();
    }
}

// ═══════════════════════════════════════════════════════════════
//  SETUP (runs once on boot)
// ═══════════════════════════════════════════════════════════════
void setup() {
    Serial.begin(115200);
    delay(1000);

    pinMode(SENSOR_POWER_PIN, OUTPUT);
    digitalWrite(SENSOR_POWER_PIN, LOW);
    analogReadResolution(12);
    pinMode(LIGHT_PIN, INPUT);

    // ── Init OLED ────────────────────────────────────────────
    Wire.begin(21, 22);
    if (!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
        Serial.println("OLED not found!");
        while (1) delay(10);
    }

    // ── Boot screen ──────────────────────────────────────────
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 15);
    display.println("  WeatherMind");
    display.setCursor(0, 30);
    display.println(" by Shadow Mechanics");
    display.display();
    delay(3000);

    // ── Init ESP-NOW ─────────────────────────────────────────
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();

    if (esp_now_init() != ESP_OK) {
        Serial.println("ESP-NOW init failed!");
    } else {
        esp_now_register_send_cb(onDataSent);
        esp_now_register_recv_cb(onDataRecv);

        esp_now_peer_info_t peerInfo = {};
        memcpy(peerInfo.peer_addr, peerAddress, 6);
        peerInfo.channel = 0;
        peerInfo.encrypt = false;
        esp_now_add_peer(&peerInfo);

        Serial.println("ESP-NOW: Ready");
        Serial.printf("ESP-NOW: Device name '%s'\n", DEVICE_NAME);
    }

    // Print MAC address for pairing
    Serial.printf("MAC: %s\n", WiFi.macAddress().c_str());

    // ── Init BLE (stays on forever) ──────────────────────────
    BLEDevice::init("WeatherMind");
    pServer = BLEDevice::createServer();
    pServer->setCallbacks(new ServerCallbacks());

    BLEService* pService = pServer->createService(SERVICE_UUID);

    pCurrentChar = pService->createCharacteristic(
        CHAR_CURRENT_UUID,
        BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY);
    pCurrentChar->addDescriptor(new BLE2902());

    pPredictedChar = pService->createCharacteristic(
        CHAR_PREDICTED_UUID,
        BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY);
    pPredictedChar->addDescriptor(new BLE2902());

    pStatusChar = pService->createCharacteristic(
        CHAR_WEATHER_STATUS_UUID,
        BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY);
    pStatusChar->addDescriptor(new BLE2902());

    pAlertChar = pService->createCharacteristic(
        CHAR_ALERT_UUID,
        BLECharacteristic::PROPERTY_NOTIFY);
    pAlertChar->addDescriptor(new BLE2902());

    pService->start();

    BLEAdvertising* pAdv = BLEDevice::getAdvertising();
    pAdv->addServiceUUID(SERVICE_UUID);
    pAdv->setScanResponse(true);
    pAdv->setMinPreferred(0x06);
    BLEDevice::startAdvertising();

    Serial.println("BLE: Always on, advertising as 'WeatherMind'");

    // ── Run first cycle immediately ──────────────────────────
    runCycle();
}

// ═══════════════════════════════════════════════════════════════
//  LOOP (light sleep between cycles, BLE stays alive)
// ═══════════════════════════════════════════════════════════════
void loop() {
    // Handle BLE reconnection
    if (!deviceConnected && oldDeviceConnected) {
        delay(500);
        BLEDevice::startAdvertising();
        Serial.println("BLE: Restarted advertising");
    }
    oldDeviceConnected = deviceConnected;

    // Handle incoming ESP-NOW messages
    if (newMessageReceived) {
        newMessageReceived = false;

        if (incomingMsg.msgType == MSG_TYPE_WEATHER) {
            Serial.printf("ESP-NOW from %s: %s [sev:%d] T:%.1fC\n",
                          incomingMsg.senderName,
                          incomingMsg.weatherName,
                          incomingMsg.severity,
                          incomingMsg.temperature);
        } else {
            Serial.printf("ESP-NOW msg from %s: %s\n",
                          incomingMsg.senderName,
                          incomingMsg.textMsg);
        }

        showIncomingMessage();
    }

    // Light sleep — BLE stays alive
    Serial.printf("Light sleep %d min...\n", SLEEP_MINUTES);
    esp_sleep_enable_timer_wakeup((uint64_t)SLEEP_MINUTES * 60ULL * 1000000ULL);
    esp_light_sleep_start();

    Serial.println("Woke from light sleep");
    runCycle();
}
