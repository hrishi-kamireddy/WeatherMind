/*
 * ═══════════════════════════════════════════════════════════════
 * WeatherMind — ESP32 NN Predictor + Weather Classifier + BLE
 * ═══════════════════════════════════════════════════════════════
 *
 * - Sensors sample every 5 seconds (fills buffer + pressure trend)
 * - NN runs every 30 MINUTES (not every sample)
 * - Classifies predicted weather into named types
 * - Sends results + alerts over BLE
 *
 * NN: 48 -> 16 (ReLU) -> 8 (ReLU) -> 4 (Sigmoid)
 *
 * Wiring (I2C, BOTH sensors share same 4 wires):
 *   BME280 VCC  + BH1750 VCC  -> ESP32 3.3V pin (NOT 5V!)
 *   BME280 GND  + BH1750 GND  -> ESP32 GND
 *   BME280 SDA  + BH1750 SDA  -> ESP32 GPIO 21
 *   BME280 SCL  + BH1750 SCL  -> ESP32 GPIO 22
 * ═══════════════════════════════════════════════════════════════
 */

#include "nn_weights.h"

#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ─── BLE UUIDs ───────────────────────────────────────────────
#define SERVICE_UUID              "e3a1f0b0-1234-5678-abcd-000000000001"
#define CHAR_CURRENT_UUID         "e3a1f0b0-1234-5678-abcd-000000000002"
#define CHAR_PREDICTED_UUID       "e3a1f0b0-1234-5678-abcd-000000000003"
#define CHAR_WEATHER_STATUS_UUID  "e3a1f0b0-1234-5678-abcd-000000000004"
#define CHAR_ALERT_UUID           "e3a1f0b0-1234-5678-abcd-000000000005"

BLEServer*         pServer        = nullptr;
BLECharacteristic* pCurrentChar   = nullptr;
BLECharacteristic* pPredictedChar = nullptr;
BLECharacteristic* pStatusChar    = nullptr;
BLECharacteristic* pAlertChar     = nullptr;
bool deviceConnected    = false;
bool oldDeviceConnected = false;

// ─── Sensors (uncomment for real hardware) ───────────────────
// #include <Wire.h>
// #include <Adafruit_BME280.h>
// #include <BH1750.h>
// Adafruit_BME280 bme;
// BH1750 lightMeter;

// ─── Timing ──────────────────────────────────────────────────
const unsigned long SAMPLE_INTERVAL_MS    = 5000;         // 5 seconds
const unsigned long INFERENCE_INTERVAL_MS = 30UL * 60000; // 30 minutes

unsigned long lastSampleTime    = 0;
unsigned long lastInferenceTime = 0;
bool          firstInferenceDone = false;

// ─── NN ring buffer ──────────────────────────────────────────
float sensorBuffer[NN_LOOKBACK][4];
int   bufferIndex = 0;
int   bufferCount = 0;

// ─── NN inference buffers ────────────────────────────────────
float inputVec[NN_INPUT_DIM];
float hidden1Buf[NN_HIDDEN1];
float hidden2Buf[NN_HIDDEN2];
float outputBuf[NN_OUTPUT_DIM];

// ─── Last prediction (displayed between inference runs) ──────
float lastPredTemp = 0, lastPredHum = 0, lastPredPres = 0, lastPredLux = 0;
String lastWeatherName   = "UNKNOWN";
String lastWeatherDetail = "";
int    lastSeverity      = 0;
bool   hasPrediction     = false;

// ─── Pressure trend tracking ─────────────────────────────────
#define PRESSURE_HISTORY_LEN 60  // 5 min at 5s intervals
float pressureHistory[PRESSURE_HISTORY_LEN];
int   pressHistIdx   = 0;
int   pressHistCount = 0;

// ─── Previous alert type (only notify on change) ─────────────
String prevAlertType = "UNKNOWN";

// ═══════════════════════════════════════════════════════════════
//  Weather Classification
//  Priority: most severe first. First match wins.
//  Pressure: table uses mb, BME280 outputs Pa. 1 mb = 100 Pa.
// ═══════════════════════════════════════════════════════════════

struct WeatherResult {
    String name;
    String detail;
    int    severity;  // 0=clear, 1=mild, 2=moderate, 3=severe
};

WeatherResult classifyWeather(float tempC, float humPct, float pressPa, float lux) {
    float pressMb = pressPa / 100.0;

    // 1. HURRICANE: >26.7C, <980mb, hum 90-100%, lux <50
    if (tempC > 26.7 && pressMb < 980.0 && humPct >= 90.0 && lux < 50.0)
        return {"HURRICANE", "Extreme low pressure + high humidity + warm", 3};

    // 2. BLIZZARD: <-7C, <995mb, hum >80%, lux <100
    if (tempC < -7.0 && pressMb < 995.0 && humPct > 80.0 && lux < 100.0)
        return {"BLIZZARD", "Extreme cold + low pressure + low visibility", 3};

    // 3. THUNDERSTORM: >18C, <1000mb, hum >70%, lux 100-1000
    if (tempC > 18.0 && pressMb < 1000.0 && humPct > 70.0 && lux >= 100.0 && lux <= 1000.0)
        return {"THUNDERSTORM", "Low pressure + warm + humid + dim skies", 3};

    // 4. SNOW: <0C, <1010mb, hum >70%, lux 1000-5000
    if (tempC < 0.0 && pressMb < 1010.0 && humPct > 70.0 && lux >= 1000.0 && lux <= 5000.0)
        return {"SNOW", "Below freezing + humid + overcast", 2};

    // 5. RAIN: 4.4-26.7C, 990-1005mb, hum 60-100%, lux 500-2000
    if (tempC >= 4.4 && tempC <= 26.7 && pressMb >= 990.0 && pressMb <= 1005.0 &&
        humPct >= 60.0 && lux >= 500.0 && lux <= 2000.0)
        return {"RAIN", "Moderate pressure drop + humid + overcast", 2};

    // 6. FOG: hum >=95%, >1013mb, lux <100
    if (humPct >= 95.0 && pressMb > 1013.0 && lux < 100.0)
        return {"FOG", "Near-saturation humidity + low visibility", 1};

    // 7. HEAT WAVE: >38C, hum <40%, lux >500
    if (tempC > 38.0 && humPct < 40.0 && lux > 500.0)
        return {"HEAT WAVE", "Extreme heat + dry + bright sun", 2};

    // 8. CLEAR: nothing matched
    return {"CLEAR", "No severe weather detected", 0};
}

// ═══════════════════════════════════════════════════════════════
//  BLE callbacks
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
//  NN math (weights live in PROGMEM flash)
// ═══════════════════════════════════════════════════════════════
static inline float pgm_float(const float* addr) {
    float v; memcpy_P(&v, addr, sizeof(float)); return v;
}
static inline float relu(float x)    { return x > 0.0f ? x : 0.0f; }
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

float normalize(float raw, int i)   { return (raw - pgm_float(&FEAT_MIN[i])) / pgm_float(&FEAT_RANGE[i]); }
float denormalize(float n, int i)   { return n * pgm_float(&FEAT_RANGE[i]) + pgm_float(&FEAT_MIN[i]); }

// ═══════════════════════════════════════════════════════════════
//  Read sensors
// ═══════════════════════════════════════════════════════════════
void readSensors(float* temp, float* hum, float* pres, float* lux) {
    // ── UNCOMMENT FOR REAL SENSORS ──
    // *temp = bme.readTemperature();
    // *hum  = bme.readHumidity();
    // *pres = bme.readPressure();
    // *lux  = lightMeter.readLightLevel();

    // ── DUMMY DATA FOR TESTING (remove when using real sensors) ──
    *temp = 24.0 + random(-10, 10) * 0.1;
    *hum  = 34.0 + random(-20, 20) * 0.1;
    *pres = 98489.0 + random(-50, 50);
    *lux  = 15.0 + random(0, 20);
}

// ═══════════════════════════════════════════════════════════════
//  Send data over BLE
// ═══════════════════════════════════════════════════════════════
void sendBLE(float curT, float curH, float curP, float curL) {
    if (!deviceConnected) return;

    char buf[80];

    // Current values
    snprintf(buf, sizeof(buf), "%.1f,%.1f,%.0f,%.1f", curT, curH, curP, curL);
    pCurrentChar->setValue(buf);
    pCurrentChar->notify();

    if (!hasPrediction) return;

    // Predicted values
    snprintf(buf, sizeof(buf), "%.1f,%.1f,%.0f,%.1f",
             lastPredTemp, lastPredHum, lastPredPres, lastPredLux);
    pPredictedChar->setValue(buf);
    pPredictedChar->notify();

    // Weather status: "THUNDERSTORM|3|detail"
    char statusBuf[120];
    snprintf(statusBuf, sizeof(statusBuf), "%s|%d|%s",
             lastWeatherName.c_str(), lastSeverity, lastWeatherDetail.c_str());
    pStatusChar->setValue(statusBuf);
    pStatusChar->notify();

    // Alert only on weather type change
    if (lastWeatherName != prevAlertType) {
        char alertBuf[120];
        if (lastSeverity >= 2)
            snprintf(alertBuf, sizeof(alertBuf), "ALERT:%s|%s",
                     lastWeatherName.c_str(), lastWeatherDetail.c_str());
        else
            snprintf(alertBuf, sizeof(alertBuf), "OK:%s|%s",
                     lastWeatherName.c_str(), lastWeatherDetail.c_str());
        pAlertChar->setValue(alertBuf);
        pAlertChar->notify();
        Serial.printf(">>> BLE ALERT: %s\n", alertBuf);
        prevAlertType = lastWeatherName;
    }
}

// ═══════════════════════════════════════════════════════════════
//  Run NN inference + classify weather
// ═══════════════════════════════════════════════════════════════
void runInference() {
    if (bufferCount < NN_LOOKBACK) return;

    // Flatten ring buffer oldest-first
    int start = bufferIndex;
    for (int s = 0; s < NN_LOOKBACK; s++) {
        int idx = (start + s) % NN_LOOKBACK;
        for (int f = 0; f < 4; f++)
            inputVec[s * 4 + f] = sensorBuffer[idx][f];
    }

    nnPredict(inputVec, outputBuf);

    lastPredTemp = denormalize(outputBuf[0], 0);
    lastPredHum  = denormalize(outputBuf[1], 1);
    lastPredPres = denormalize(outputBuf[2], 2);
    lastPredLux  = denormalize(outputBuf[3], 3);
    hasPrediction = true;

    WeatherResult w = classifyWeather(lastPredTemp, lastPredHum, lastPredPres, lastPredLux);
    lastWeatherName   = w.name;
    lastWeatherDetail = w.detail;
    lastSeverity      = w.severity;

    const char* sevLabel[] = {"OK", "MILD", "MODERATE", "SEVERE"};
    Serial.println("────────────────────────────────────────");
    Serial.println("  NN INFERENCE (runs every 30 min)");
    Serial.printf("  PRED -> T:%.1fC  H:%.1f%%  P:%.0fPa  L:%.1flux\n",
                  lastPredTemp, lastPredHum, lastPredPres, lastPredLux);
    Serial.printf("  WEATHER -> %s [%s]: %s\n",
                  w.name.c_str(), sevLabel[w.severity], w.detail.c_str());
    Serial.println("────────────────────────────────────────");
}

// ═══════════════════════════════════════════════════════════════
//  SETUP
// ═══════════════════════════════════════════════════════════════
void setup() {
    Serial.begin(115200);
    delay(1000);

    Serial.println();
    Serial.println("=========================================");
    Serial.println("  WeatherMind - ESP32 NN + BLE");
    Serial.println("=========================================");
    Serial.printf("  NN: %d -> %d -> %d -> %d\n",
                  NN_INPUT_DIM, NN_HIDDEN1, NN_HIDDEN2, NN_OUTPUT_DIM);
    Serial.println("  Sensors: every 5s");
    Serial.println("  NN inference: every 30 min");
    Serial.println("  Weather types: HURRICANE, BLIZZARD,");
    Serial.println("    THUNDERSTORM, SNOW, RAIN, FOG,");
    Serial.println("    HEAT WAVE, CLEAR");
    Serial.println("=========================================");

    // ── Sensors (uncomment for real hardware) ──
    // Wire.begin(21, 22);  // SDA=21, SCL=22
    // if (!bme.begin(0x76)) {
    //     Serial.println("ERROR: BME280 not found! Check wiring (3.3V, not 5V)");
    //     while (1) delay(10);
    // }
    // if (!lightMeter.begin()) {
    //     Serial.println("ERROR: BH1750 not found! Check wiring (3.3V, not 5V)");
    //     while (1) delay(10);
    // }
    // Serial.println("Sensors OK (3.3V I2C on GPIO 21/22)");

    // ── BLE ──
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
    pAdv->setMinPreferred(0x12);
    BLEDevice::startAdvertising();

    Serial.println("BLE: Advertising as 'WeatherMind'");
    Serial.println("Filling sensor buffer...\n");
}

// ═══════════════════════════════════════════════════════════════
//  LOOP
// ═══════════════════════════════════════════════════════════════
void loop() {
    unsigned long now = millis();

    // ── Sample sensors every 5 seconds ───────────────────────
    if (now - lastSampleTime < SAMPLE_INTERVAL_MS) return;
    lastSampleTime = now;

    float temp, hum, pres, lx;
    readSensors(&temp, &hum, &pres, &lx);

    // Track pressure history for trend detection
    pressureHistory[pressHistIdx] = pres;
    pressHistIdx = (pressHistIdx + 1) % PRESSURE_HISTORY_LEN;
    if (pressHistCount < PRESSURE_HISTORY_LEN) pressHistCount++;

    // Store normalized values in NN ring buffer
    sensorBuffer[bufferIndex][0] = normalize(temp, 0);
    sensorBuffer[bufferIndex][1] = normalize(hum,  1);
    sensorBuffer[bufferIndex][2] = normalize(pres, 2);
    sensorBuffer[bufferIndex][3] = normalize(lx,   3);
    bufferIndex = (bufferIndex + 1) % NN_LOOKBACK;
    if (bufferCount < NN_LOOKBACK) bufferCount++;

    // Print current reading
    Serial.printf("NOW  -> T:%.1fC  H:%.1f%%  P:%.0fPa  L:%.1flux",
                  temp, hum, pres, lx);

    if (hasPrediction) {
        Serial.printf("  |  PRED: %s [sev:%d]",
                      lastWeatherName.c_str(), lastSeverity);
    } else if (bufferCount < NN_LOOKBACK) {
        Serial.printf("  |  Buffer: %d/%d", bufferCount, NN_LOOKBACK);
    }
    Serial.println();

    // ── Run NN every 30 minutes ──────────────────────────────
    // Also run once as soon as buffer is full (first inference)
    bool bufferReady = (bufferCount >= NN_LOOKBACK);
    bool timeForInference = (now - lastInferenceTime >= INFERENCE_INTERVAL_MS);
    bool needFirstRun = (bufferReady && !firstInferenceDone);

    if (bufferReady && (timeForInference || needFirstRun)) {
        runInference();
        lastInferenceTime = now;
        firstInferenceDone = true;
    }

    // ── Send current data + last prediction over BLE ─────────
    sendBLE(temp, hum, pres, lx);

    // ── Handle BLE reconnection ──────────────────────────────
    if (!deviceConnected && oldDeviceConnected) {
        delay(500);
        BLEDevice::startAdvertising();
        Serial.println("BLE: Restarted advertising");
    }
    oldDeviceConnected = deviceConnected;
}
