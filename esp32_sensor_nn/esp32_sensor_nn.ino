/*
 * ═══════════════════════════════════════════════════════════════
 * WeatherMind — ESP32 NN Predictor + BLE (Light Sleep)
 * ═══════════════════════════════════════════════════════════════
 *
 * Light sleep mode: ESP32 rests at ~2-5mA between cycles.
 * BLE stays alive — phone can connect anytime.
 * OLED stays on (powered by 3.3V).
 *
 * Each cycle:
 *   1. Wake from light sleep
 *   2. Power on sensors via GPIO 15
 *   3. Collect 12 readings over ~1 minute (5s intervals)
 *   4. Run NN -> predict 30 min ahead -> classify weather
 *   5. Update OLED + notify connected BLE devices
 *   6. Power off sensors
 *   7. Light sleep (BLE stays on, OLED stays on)
 *
 * CYCLE TIME:
 *   Currently set to 1 minute sleep for testing.
 *   For production, change SLEEP_MINUTES from 1 to 29.
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

// ─── Sensor objects ──────────────────────────────────────────
DHT dht(DHT_PIN, DHT11);
Adafruit_BMP085 bmp;
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// ─── BLE objects (persist across light sleep) ────────────────
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
//  NN math
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
//  Read sensors
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
//  Update OLED (stays visible during sleep)
// ═══════════════════════════════════════════════════════════════
void updateOLED() {
    display.clearDisplay();
    display.setTextColor(SSD1306_WHITE);

    // ── Current values ───────────────────────────
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

    // ── Separator ────────────────────────────────
    display.drawLine(0, 30, 127, 30, SSD1306_WHITE);

    // ── Forecast ─────────────────────────────────
    display.setTextSize(1);
    display.setCursor(0, 34);
    display.print("30m: ");
    display.print(predTemp, 1);
    display.print("C");

    // Weather type large
    display.setTextSize(2);
    display.setCursor(0, 46);
    display.print(weatherName);

    display.display();
}

// ═══════════════════════════════════════════════════════════════
//  Update BLE characteristics
// ═══════════════════════════════════════════════════════════════
void updateBLE() {
    char buf[80];

    // Current values
    snprintf(buf, sizeof(buf), "%.1f,%.1f,%.0f,%.0f", curTemp, curHum, curPres, curLux);
    pCurrentChar->setValue(buf);
    if (deviceConnected) pCurrentChar->notify();

    if (!hasPrediction) return;

    // Predicted values
    snprintf(buf, sizeof(buf), "%.1f,%.1f,%.0f,%.1f", predTemp, predHum, predPres, predLux);
    pPredictedChar->setValue(buf);
    if (deviceConnected) pPredictedChar->notify();

    // Weather status
    char statusBuf[120];
    snprintf(statusBuf, sizeof(statusBuf), "%s|%d|%s", weatherName, weatherSev, weatherDetail);
    pStatusChar->setValue(statusBuf);
    if (deviceConnected) pStatusChar->notify();

    // Alert on weather change
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
//  Collect data + run NN + update everything
// ═══════════════════════════════════════════════════════════════
void runCycle() {
    cycleCount++;
    Serial.printf("\n=========================================\n");
    Serial.printf("  WeatherMind - cycle #%d\n", cycleCount);
    Serial.printf("=========================================\n");

    // ── Power on sensors ─────────────────────────────────────
    digitalWrite(SENSOR_POWER_PIN, HIGH);
    delay(3000);

    // ── Re-init sensors (they were powered off) ──────────────
    Wire.begin(21, 22);
    dht.begin();

    if (!bmp.begin()) {
        Serial.println("BMP180 not found!");
        return;
    }

    // ── Show collecting on OLED ──────────────────────────────
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

    // ── Update OLED + BLE ────────────────────────────────────
    updateOLED();
    updateBLE();
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
//  LOOP (light sleep between cycles)
// ═══════════════════════════════════════════════════════════════
void loop() {
    // Handle BLE reconnection
    if (!deviceConnected && oldDeviceConnected) {
        delay(500);
        BLEDevice::startAdvertising();
        Serial.println("BLE: Restarted advertising");
    }
    oldDeviceConnected = deviceConnected;

    // Light sleep for SLEEP_MINUTES
    // esp_sleep_enable_timer_wakeup keeps BLE alive
    Serial.printf("Light sleep %d min (BLE stays on)...\n", SLEEP_MINUTES);
    esp_sleep_enable_timer_wakeup((uint64_t)SLEEP_MINUTES * 60ULL * 1000000ULL);
    esp_light_sleep_start();

    // Woke up — run next cycle
    Serial.println("Woke from light sleep");
    runCycle();
}
