/*
 * ═══════════════════════════════════════════════════════════════
 * WeatherMind — Tiny NN Sensor Predictor + BLE Alerts (ESP32)
 * ═══════════════════════════════════════════════════════════════
 *
 * Reads temperature, humidity, pressure, and light (lux) from
 * sensors, feeds them into a tiny neural network, predicts
 * values ~30 minutes ahead, classifies weather as GOOD/BAD,
 * and broadcasts everything over BLE.
 *
 * Network architecture:
 *   Input:  48 (12 timesteps × 4 features)
 *   Hidden: 16 neurons (ReLU)
 *   Hidden:  8 neurons (ReLU)
 *   Output:  4 (temperature, humidity, pressure, lux)
 *
 * Hardware:
 *   - BME280 / BME680 for temperature, humidity, pressure (I2C)
 *   - BH1750 for lux (I2C)
 *   - Built-in ESP32 BLE (no extra hardware)
 *
 * BLE Services:
 *   - Weather Service: current & predicted values + alert status
 *   - Phones connect and receive notifications on weather change
 * ═══════════════════════════════════════════════════════════════
 */

#include "nn_weights.h"

// ─── BLE ─────────────────────────────────────────────────────
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// UUIDs (generated for WeatherMind)
#define SERVICE_UUID              "e3a1f0b0-1234-5678-abcd-000000000001"
#define CHAR_CURRENT_UUID         "e3a1f0b0-1234-5678-abcd-000000000002"
#define CHAR_PREDICTED_UUID       "e3a1f0b0-1234-5678-abcd-000000000003"
#define CHAR_WEATHER_STATUS_UUID  "e3a1f0b0-1234-5678-abcd-000000000004"
#define CHAR_ALERT_UUID           "e3a1f0b0-1234-5678-abcd-000000000005"

BLEServer*         pServer         = nullptr;
BLECharacteristic* pCurrentChar    = nullptr;
BLECharacteristic* pPredictedChar  = nullptr;
BLECharacteristic* pStatusChar     = nullptr;
BLECharacteristic* pAlertChar      = nullptr;
bool deviceConnected    = false;
bool oldDeviceConnected = false;

// ─── Sensor includes (uncomment for real hardware) ───────────
// #include <Wire.h>
// #include <Adafruit_BME280.h>
// #include <BH1750.h>
// Adafruit_BME280 bme;
// BH1750 lightMeter;

// ─── Ring buffer for lookback window ─────────────────────────
float sensorBuffer[NN_LOOKBACK][4];
int   bufferIndex  = 0;
int   bufferCount  = 0;

// ─── Timing ──────────────────────────────────────────────────
const unsigned long SAMPLE_INTERVAL_MS = 5000;
unsigned long lastSampleTime = 0;

// ─── NN inference buffers ────────────────────────────────────
float inputVec[NN_INPUT_DIM];
float hidden1[NN_HIDDEN1];
float hidden2[NN_HIDDEN2];
float output[NN_OUTPUT_DIM];

// ─── Weather classification ──────────────────────────────────
// Track pressure history for trend detection
#define PRESSURE_HISTORY_LEN 60   // 5 min of readings at 5s intervals
float pressureHistory[PRESSURE_HISTORY_LEN];
int   pressHistIdx   = 0;
int   pressHistCount = 0;

// Previous alert state (to only notify on change)
String lastAlertLevel = "UNKNOWN";

// ═══════════════════════════════════════════════════════════════
//  Weather Classification Thresholds
//  Adjust these for your local climate!
// ═══════════════════════════════════════════════════════════════
struct WeatherThresholds {
    // Temperature (°C)
    float tempHigh       = 38.0;   // dangerously hot
    float tempLow        =  2.0;   // near freezing
    float tempWarnHigh   = 35.0;   // uncomfortable heat
    float tempWarnLow    =  5.0;   // cold warning

    // Humidity (%)
    float humHigh        = 85.0;   // likely rain / storm
    float humWarnHigh    = 75.0;   // muggy, possible rain

    // Pressure (Pa) — drop rate over 5 minutes
    float pressDropBad   = -300.0; // rapid drop = storm
    float pressDropWarn  = -150.0; // moderate drop = weather change

    // Lux
    float luxDark        =  5.0;   // very dark (heavy clouds or night)
} thresholds;

// Weather status enum
enum WeatherLevel {
    WEATHER_GOOD,
    WEATHER_FAIR,
    WEATHER_WARNING,
    WEATHER_BAD
};

struct WeatherResult {
    WeatherLevel level;
    String       label;    // "GOOD", "FAIR", "WARNING", "BAD"
    String       reason;   // human-readable explanation
};

// ═══════════════════════════════════════════════════════════════
//  Classify weather from predicted values
// ═══════════════════════════════════════════════════════════════
WeatherResult classifyWeather(float temp, float hum, float pres, float lux) {
    WeatherResult result;
    result.level = WEATHER_GOOD;
    result.label = "GOOD";
    result.reason = "Clear conditions";

    // Calculate pressure trend (Pa change over last 5 minutes)
    float pressTrend = 0.0;
    if (pressHistCount >= PRESSURE_HISTORY_LEN) {
        int oldestIdx = pressHistIdx;  // ring buffer oldest
        float oldPressure = pressureHistory[oldestIdx];
        pressTrend = pres - oldPressure;
    }

    // ── Check BAD conditions first (most severe) ──
    if (temp >= thresholds.tempHigh) {
        result.level = WEATHER_BAD;
        result.label = "BAD";
        result.reason = "Extreme heat predicted";
    }
    else if (temp <= thresholds.tempLow) {
        result.level = WEATHER_BAD;
        result.label = "BAD";
        result.reason = "Near-freezing temperatures";
    }
    else if (hum >= thresholds.humHigh && pressTrend <= thresholds.pressDropBad) {
        result.level = WEATHER_BAD;
        result.label = "BAD";
        result.reason = "Storm likely: high humidity + pressure drop";
    }
    else if (pressTrend <= thresholds.pressDropBad) {
        result.level = WEATHER_BAD;
        result.label = "BAD";
        result.reason = "Rapid pressure drop: storm approaching";
    }
    // ── WARNING conditions ──
    else if (temp >= thresholds.tempWarnHigh) {
        result.level = WEATHER_WARNING;
        result.label = "WARNING";
        result.reason = "High heat expected";
    }
    else if (temp <= thresholds.tempWarnLow) {
        result.level = WEATHER_WARNING;
        result.label = "WARNING";
        result.reason = "Cold conditions expected";
    }
    else if (hum >= thresholds.humWarnHigh) {
        result.level = WEATHER_WARNING;
        result.label = "WARNING";
        result.reason = "High humidity: possible rain";
    }
    else if (pressTrend <= thresholds.pressDropWarn) {
        result.level = WEATHER_WARNING;
        result.label = "WARNING";
        result.reason = "Pressure dropping: weather changing";
    }
    // ── FAIR conditions ──
    else if (hum >= 65.0 || lux <= thresholds.luxDark) {
        result.level = WEATHER_FAIR;
        result.label = "FAIR";
        result.reason = "Partly cloudy or overcast";
    }

    return result;
}

// ═══════════════════════════════════════════════════════════════
//  BLE server callbacks
// ═══════════════════════════════════════════════════════════════
class ServerCallbacks : public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) override {
        deviceConnected = true;
        Serial.println("BLE: Device connected");
    }
    void onDisconnect(BLEServer* pServer) override {
        deviceConnected = false;
        Serial.println("BLE: Device disconnected");
    }
};

// ═══════════════════════════════════════════════════════════════
//  NN helper functions
// ═══════════════════════════════════════════════════════════════
static inline float pgm_read_float_near(const float* addr) {
    float val;
    memcpy_P(&val, addr, sizeof(float));
    return val;
}

static inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

void denseForward(
    const float* input,  int inSize,
    float* output,       int outSize,
    const float* weights,
    const float* biases,
    char activation
) {
    for (int j = 0; j < outSize; j++) {
        float sum = pgm_read_float_near(&biases[j]);
        for (int i = 0; i < inSize; i++) {
            float w = pgm_read_float_near(&weights[i * outSize + j]);
            sum += input[i] * w;
        }
        if (activation == 'r')      sum = relu(sum);
        else if (activation == 's')  sum = sigmoid(sum);
        output[j] = sum;
    }
}

void nnPredict(float* inputVec, float* result) {
    denseForward(inputVec, NN_INPUT_DIM, hidden1, NN_HIDDEN1,
                 W_HIDDEN1, B_HIDDEN1, 'r');
    denseForward(hidden1, NN_HIDDEN1, hidden2, NN_HIDDEN2,
                 W_HIDDEN2, B_HIDDEN2, 'r');
    denseForward(hidden2, NN_HIDDEN2, result, NN_OUTPUT_DIM,
                 W_OUTPUT, B_OUTPUT, 's');
}

float normalize(float raw, int featureIdx) {
    float mn  = pgm_read_float_near(&FEAT_MIN[featureIdx]);
    float rng = pgm_read_float_near(&FEAT_RANGE[featureIdx]);
    return (raw - mn) / rng;
}

float denormalize(float norm, int featureIdx) {
    float mn  = pgm_read_float_near(&FEAT_MIN[featureIdx]);
    float rng = pgm_read_float_near(&FEAT_RANGE[featureIdx]);
    return norm * rng + mn;
}

// ═══════════════════════════════════════════════════════════════
//  Read sensors (REPLACE with your actual sensor code)
// ═══════════════════════════════════════════════════════════════
void readSensors(float* temperature, float* humidity, float* pressure, float* lux) {
    // ── Uncomment for real sensors ──
    // *temperature = bme.readTemperature();
    // *humidity    = bme.readHumidity();
    // *pressure    = bme.readPressure();
    // *lux         = lightMeter.readLightLevel();

    // Placeholder: dummy data for testing
    *temperature = 24.0 + random(-10, 10) * 0.1;
    *humidity    = 34.0 + random(-20, 20) * 0.1;
    *pressure    = 98489.0 + random(-50, 50);
    *lux         = 15.0 + random(0, 20);
}

// ═══════════════════════════════════════════════════════════════
//  Update BLE characteristics
// ═══════════════════════════════════════════════════════════════
void updateBLE(float curT, float curH, float curP, float curL,
               float predT, float predH, float predP, float predL,
               WeatherResult& weather) {

    if (!deviceConnected) return;

    // Current values as comma-separated string
    char currentStr[80];
    snprintf(currentStr, sizeof(currentStr),
             "%.1f,%.1f,%.0f,%.1f", curT, curH, curP, curL);
    pCurrentChar->setValue(currentStr);
    pCurrentChar->notify();

    // Predicted values
    char predictedStr[80];
    snprintf(predictedStr, sizeof(predictedStr),
             "%.1f,%.1f,%.0f,%.1f", predT, predH, predP, predL);
    pPredictedChar->setValue(predictedStr);
    pPredictedChar->notify();

    // Weather status: "GOOD|Clear conditions"
    String statusStr = weather.label + "|" + weather.reason;
    pStatusChar->setValue(statusStr.c_str());
    pStatusChar->notify();

    // Alert: only notify on status CHANGE
    if (weather.label != lastAlertLevel) {
        String alertMsg;
        if (weather.level >= WEATHER_WARNING) {
            alertMsg = "ALERT:" + weather.label + "|" + weather.reason;
        } else {
            alertMsg = "OK:" + weather.label + "|" + weather.reason;
        }
        pAlertChar->setValue(alertMsg.c_str());
        pAlertChar->notify();

        Serial.printf("BLE ALERT SENT: %s\n", alertMsg.c_str());
        lastAlertLevel = weather.label;
    }
}

// ═══════════════════════════════════════════════════════════════
//  SETUP
// ═══════════════════════════════════════════════════════════════
void setup() {
    Serial.begin(115200);
    delay(1000);

    Serial.println();
    Serial.println("══════════════════════════════════════════");
    Serial.println("  WeatherMind — ESP32 NN + BLE");
    Serial.println("══════════════════════════════════════════");
    Serial.printf("  NN: %d -> %d -> %d -> %d\n",
                  NN_INPUT_DIM, NN_HIDDEN1, NN_HIDDEN2, NN_OUTPUT_DIM);
    Serial.println("══════════════════════════════════════════");

    // ── Initialize sensors (uncomment for real hardware) ──
    // Wire.begin();
    // if (!bme.begin(0x76)) {
    //     Serial.println("BME280 not found!");
    //     while (1) delay(10);
    // }
    // if (!lightMeter.begin()) {
    //     Serial.println("BH1750 not found!");
    //     while (1) delay(10);
    // }

    // ── Initialize BLE ──
    BLEDevice::init("WeatherMind");
    pServer = BLEDevice::createServer();
    pServer->setCallbacks(new ServerCallbacks());

    BLEService* pService = pServer->createService(SERVICE_UUID);

    // Current sensor values (read + notify)
    pCurrentChar = pService->createCharacteristic(
        CHAR_CURRENT_UUID,
        BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY
    );
    pCurrentChar->addDescriptor(new BLE2902());

    // Predicted values (read + notify)
    pPredictedChar = pService->createCharacteristic(
        CHAR_PREDICTED_UUID,
        BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY
    );
    pPredictedChar->addDescriptor(new BLE2902());

    // Weather status string (read + notify)
    pStatusChar = pService->createCharacteristic(
        CHAR_WEATHER_STATUS_UUID,
        BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY
    );
    pStatusChar->addDescriptor(new BLE2902());

    // Alert characteristic (notify only — fires on status change)
    pAlertChar = pService->createCharacteristic(
        CHAR_ALERT_UUID,
        BLECharacteristic::PROPERTY_NOTIFY
    );
    pAlertChar->addDescriptor(new BLE2902());

    pService->start();

    // Start advertising
    BLEAdvertising* pAdvertising = BLEDevice::getAdvertising();
    pAdvertising->addServiceUUID(SERVICE_UUID);
    pAdvertising->setScanResponse(true);
    pAdvertising->setMinPreferred(0x06);
    pAdvertising->setMinPreferred(0x12);
    BLEDevice::startAdvertising();

    Serial.println("BLE: Advertising as 'WeatherMind'");
    Serial.println("Filling sensor buffer...\n");
}

// ═══════════════════════════════════════════════════════════════
//  LOOP
// ═══════════════════════════════════════════════════════════════
void loop() {
    unsigned long now = millis();
    if (now - lastSampleTime < SAMPLE_INTERVAL_MS) return;
    lastSampleTime = now;

    // 1. Read sensors
    float temp, hum, pres, lx;
    readSensors(&temp, &hum, &pres, &lx);

    // 2. Track pressure history for trend detection
    pressureHistory[pressHistIdx] = pres;
    pressHistIdx = (pressHistIdx + 1) % PRESSURE_HISTORY_LEN;
    if (pressHistCount < PRESSURE_HISTORY_LEN) pressHistCount++;

    // 3. Normalize and store in NN ring buffer
    sensorBuffer[bufferIndex][0] = normalize(temp, 0);
    sensorBuffer[bufferIndex][1] = normalize(hum,  1);
    sensorBuffer[bufferIndex][2] = normalize(pres, 2);
    sensorBuffer[bufferIndex][3] = normalize(lx,   3);
    bufferIndex = (bufferIndex + 1) % NN_LOOKBACK;
    if (bufferCount < NN_LOOKBACK) bufferCount++;

    Serial.printf("NOW  -> T:%.1fC  H:%.1f%%  P:%.0fPa  L:%.1flux\n",
                  temp, hum, pres, lx);

    // 4. Run inference once buffer is full
    if (bufferCount >= NN_LOOKBACK) {
        // Flatten ring buffer (oldest first)
        int start = bufferIndex;
        for (int s = 0; s < NN_LOOKBACK; s++) {
            int idx = (start + s) % NN_LOOKBACK;
            for (int f = 0; f < 4; f++) {
                inputVec[s * 4 + f] = sensorBuffer[idx][f];
            }
        }

        // Run NN
        nnPredict(inputVec, output);

        float predTemp = denormalize(output[0], 0);
        float predHum  = denormalize(output[1], 1);
        float predPres = denormalize(output[2], 2);
        float predLux  = denormalize(output[3], 3);

        Serial.printf("PRED -> T:%.1fC  H:%.1f%%  P:%.0fPa  L:%.1flux\n",
                      predTemp, predHum, predPres, predLux);

        // 5. Classify weather based on PREDICTED values
        WeatherResult weather = classifyWeather(predTemp, predHum, predPres, predLux);
        Serial.printf("STATUS -> %s: %s\n",
                      weather.label.c_str(), weather.reason.c_str());

        // 6. Send over BLE
        updateBLE(temp, hum, pres, lx,
                  predTemp, predHum, predPres, predLux,
                  weather);
    } else {
        Serial.printf("Buffer: %d/%d readings\n", bufferCount, NN_LOOKBACK);
    }

    // Handle BLE reconnection
    if (!deviceConnected && oldDeviceConnected) {
        delay(500);
        BLEDevice::startAdvertising();
        Serial.println("BLE: Restarted advertising");
    }
    oldDeviceConnected = deviceConnected;

    Serial.println();
}
