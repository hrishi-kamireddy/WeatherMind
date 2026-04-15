/*
 * ═══════════════════════════════════════════════════════════════
 * WeatherMind — ESP32 NN Predictor + Weather Classifier + BLE
 * ═══════════════════════════════════════════════════════════════
 *
 * Hardware (YOUR actual setup):
 *   - DHT11 on GPIO 4 (temperature + humidity)
 *   - BMP180 on I2C (pressure)
 *   - Analog light sensor on GPIO 2
 *   - SSD1306 OLED 128x64 on I2C (display)
 *   - Sensor power controlled via GPIO 15
 *   - Deep sleep between readings
 *
 * Wiring:
 *   DHT11 data    -> GPIO 4
 *   BMP180 SDA    -> GPIO 21
 *   BMP180 SCL    -> GPIO 22
 *   OLED SDA      -> GPIO 21 (shared I2C bus)
 *   OLED SCL      -> GPIO 22 (shared I2C bus)
 *   Light sensor   -> GPIO 2 (analog)
 *   Sensor VCC     -> GPIO 15 (power control)
 *
 * Deep sleep note:
 *   Deep sleep reboots the ESP32 each cycle, wiping RAM.
 *   The NN ring buffer and state are stored in RTC memory
 *   so they survive across sleep cycles.
 *
 * NN: 48 -> 16 (ReLU) -> 8 (ReLU) -> 4 (Sigmoid)
 * Inference runs every 30 minutes (every ~360 wake cycles at 5s,
 *   or every 1 cycle if SLEEP_SECONDS=1800)
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
#define SLEEP_SECONDS    10    // Change to 1800 for 30 min production
#define WAKE_SECONDS     15    // How long OLED stays on
#define INFERENCE_EVERY  360   // Run NN every N wake cycles
                               // At 5s sleep: 360 * 5s = 30 min
                               // At 1800s sleep: set to 1

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

// ═══════════════════════════════════════════════════════════════
//  RTC Memory — survives deep sleep
//  (RAM is wiped on every deep sleep reboot, RTC memory is not)
// ═══════════════════════════════════════════════════════════════
RTC_DATA_ATTR float    sensorBuffer[NN_LOOKBACK][4];
RTC_DATA_ATTR int      bufferIndex  = 0;
RTC_DATA_ATTR int      bufferCount  = 0;
RTC_DATA_ATTR int      wakeCount    = 0;
RTC_DATA_ATTR bool     hasPrediction = false;
RTC_DATA_ATTR float    lastPredTemp  = 0;
RTC_DATA_ATTR float    lastPredHum   = 0;
RTC_DATA_ATTR float    lastPredPres  = 0;
RTC_DATA_ATTR float    lastPredLux   = 0;
RTC_DATA_ATTR int      lastSeverity  = 0;
RTC_DATA_ATTR char     lastWeatherName[20]   = "UNKNOWN";
RTC_DATA_ATTR char     lastWeatherDetail[80] = "";
RTC_DATA_ATTR char     prevAlertType[20]     = "UNKNOWN";

// ─── NN inference buffers (normal RAM, used only during wake) ─
float inputVec[NN_INPUT_DIM];
float hidden1Buf[NN_HIDDEN1];
float hidden2Buf[NN_HIDDEN2];
float outputBuf[NN_OUTPUT_DIM];

// ═══════════════════════════════════════════════════════════════
//  Weather Classification
//  Priority: most severe first. First match wins.
//  Pressure: table uses mb, sensors output Pa. 1 mb = 100 Pa.
// ═══════════════════════════════════════════════════════════════

struct WeatherResult {
    const char* name;
    const char* detail;
    int         severity;  // 0=clear, 1=mild, 2=moderate, 3=severe
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

    // 8. CLEAR
    return {"CLEAR", "No severe weather detected", 0};
}

// ═══════════════════════════════════════════════════════════════
//  NN math (weights in PROGMEM flash)
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
//  Run NN inference + classify
// ═══════════════════════════════════════════════════════════════
void runInference() {
    if (bufferCount < NN_LOOKBACK) return;

    int start = bufferIndex;
    for (int s = 0; s < NN_LOOKBACK; s++) {
        int idx = (start + s) % NN_LOOKBACK;
        for (int f = 0; f < 4; f++)
            inputVec[s * 4 + f] = sensorBuffer[idx][f];
    }

    nnPredict(inputVec, outputBuf);

    lastPredTemp = denormalizeVal(outputBuf[0], 0);
    lastPredHum  = denormalizeVal(outputBuf[1], 1);
    lastPredPres = denormalizeVal(outputBuf[2], 2);
    lastPredLux  = denormalizeVal(outputBuf[3], 3);
    hasPrediction = true;

    WeatherResult w = classifyWeather(lastPredTemp, lastPredHum, lastPredPres, lastPredLux);
    strncpy(lastWeatherName, w.name, sizeof(lastWeatherName) - 1);
    strncpy(lastWeatherDetail, w.detail, sizeof(lastWeatherDetail) - 1);
    lastSeverity = w.severity;

    const char* sevLabel[] = {"OK", "MILD", "MODERATE", "SEVERE"};
    Serial.println("────────────────────────────────────────");
    Serial.println("  NN INFERENCE");
    Serial.printf("  PRED -> T:%.1fC  H:%.1f%%  P:%.0fPa  L:%.1f\n",
                  lastPredTemp, lastPredHum, lastPredPres, lastPredLux);
    Serial.printf("  WEATHER -> %s [%s]: %s\n",
                  w.name, sevLabel[w.severity], w.detail);
    Serial.println("────────────────────────────────────────");
}

// ═══════════════════════════════════════════════════════════════
//  BLE — quick broadcast then sleep
//  Since deep sleep kills BLE, we init, advertise, send one
//  burst of data, wait briefly for a connection, then sleep.
// ═══════════════════════════════════════════════════════════════
void doBLE(float curT, float curH, float curP, float curL) {
    BLEDevice::init("WeatherMind");
    BLEServer* pServer = BLEDevice::createServer();

    BLEService* pService = pServer->createService(SERVICE_UUID);

    BLECharacteristic* pCurrentChar = pService->createCharacteristic(
        CHAR_CURRENT_UUID,
        BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY);
    pCurrentChar->addDescriptor(new BLE2902());

    BLECharacteristic* pPredictedChar = pService->createCharacteristic(
        CHAR_PREDICTED_UUID,
        BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY);
    pPredictedChar->addDescriptor(new BLE2902());

    BLECharacteristic* pStatusChar = pService->createCharacteristic(
        CHAR_WEATHER_STATUS_UUID,
        BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY);
    pStatusChar->addDescriptor(new BLE2902());

    BLECharacteristic* pAlertChar = pService->createCharacteristic(
        CHAR_ALERT_UUID,
        BLECharacteristic::PROPERTY_NOTIFY);
    pAlertChar->addDescriptor(new BLE2902());

    pService->start();

    BLEAdvertising* pAdv = BLEDevice::getAdvertising();
    pAdv->addServiceUUID(SERVICE_UUID);
    pAdv->setScanResponse(true);
    BLEDevice::startAdvertising();

    // Set current values
    char buf[80];
    snprintf(buf, sizeof(buf), "%.1f,%.1f,%.0f,%.1f", curT, curH, curP, curL);
    pCurrentChar->setValue(buf);

    // Set predicted values
    if (hasPrediction) {
        snprintf(buf, sizeof(buf), "%.1f,%.1f,%.0f,%.1f",
                 lastPredTemp, lastPredHum, lastPredPres, lastPredLux);
        pPredictedChar->setValue(buf);

        char statusBuf[120];
        snprintf(statusBuf, sizeof(statusBuf), "%s|%d|%s",
                 lastWeatherName, lastSeverity, lastWeatherDetail);
        pStatusChar->setValue(statusBuf);

        // Alert on weather change
        if (strcmp(lastWeatherName, prevAlertType) != 0) {
            char alertBuf[120];
            if (lastSeverity >= 2)
                snprintf(alertBuf, sizeof(alertBuf), "ALERT:%s|%s",
                         lastWeatherName, lastWeatherDetail);
            else
                snprintf(alertBuf, sizeof(alertBuf), "OK:%s|%s",
                         lastWeatherName, lastWeatherDetail);
            pAlertChar->setValue(alertBuf);
            Serial.printf(">>> BLE ALERT: %s\n", alertBuf);
            strncpy(prevAlertType, lastWeatherName, sizeof(prevAlertType) - 1);
        }
    }

    // Keep BLE alive during OLED display time (WAKE_SECONDS)
    // Phone can connect during this window
    Serial.println("BLE: Advertising as 'WeatherMind'");
}

// ═══════════════════════════════════════════════════════════════
//  SETUP (runs every wake from deep sleep)
// ═══════════════════════════════════════════════════════════════
void setup() {
    Serial.begin(115200);
    wakeCount++;

    Serial.printf("\n=== WeatherMind wake #%d ===\n", wakeCount);

    // ── Power on sensors ─────────────────────────────────────
    pinMode(SENSOR_POWER_PIN, OUTPUT);
    digitalWrite(SENSOR_POWER_PIN, HIGH);
    delay(3000);

    // ── Init I2C and sensors ─────────────────────────────────
    Wire.begin(21, 22);
    dht.begin();
    analogReadResolution(12);
    pinMode(LIGHT_PIN, INPUT);

    if (!bmp.begin()) {
        Serial.println("BMP180 not found!");
        while (1) delay(10);
    }

    if (!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
        Serial.println("OLED not found!");
        while (1) delay(10);
    }

    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);

    // ── Read sensors (with retry for DHT11) ──────────────────
    dht.readTemperature();
    dht.readHumidity();
    delay(2500);

    float temp     = dht.readTemperature();
    float humidity = dht.readHumidity();

    if (isnan(temp) || isnan(humidity) || temp == 0) {
        delay(2000);
        temp     = dht.readTemperature();
        humidity = dht.readHumidity();
    }

    int   lightRaw = analogRead(LIGHT_PIN);
    float lightLux = (float)lightRaw;  // raw 0-4095 analog value used as lux proxy
    float pressure = bmp.readSealevelPressure(515);  // Pa, adjusted for altitude

    Serial.printf("NOW  -> T:%.1fC  H:%.1f%%  P:%.0fPa  L:%.0f\n",
                  temp, humidity, pressure, lightLux);

    // ── Store in NN ring buffer (RTC memory) ─────────────────
    sensorBuffer[bufferIndex][0] = normalizeVal(temp,     0);
    sensorBuffer[bufferIndex][1] = normalizeVal(humidity, 1);
    sensorBuffer[bufferIndex][2] = normalizeVal(pressure, 2);
    sensorBuffer[bufferIndex][3] = normalizeVal(lightLux, 3);
    bufferIndex = (bufferIndex + 1) % NN_LOOKBACK;
    if (bufferCount < NN_LOOKBACK) bufferCount++;

    // ── Run NN inference on schedule ─────────────────────────
    bool bufferReady = (bufferCount >= NN_LOOKBACK);
    bool timeToInfer = (wakeCount % INFERENCE_EVERY == 0);
    bool firstRun    = (bufferReady && !hasPrediction);

    if (bufferReady && (timeToInfer || firstRun)) {
        runInference();
    }

    // ── Start BLE (advertises during OLED wake time) ─────────
    doBLE(temp, humidity, pressure, lightLux);

    // ── OLED display ─────────────────────────────────────────
    display.clearDisplay();
    display.setCursor(0, 0);

    // Current values
    display.print("Temp:     "); display.print(temp, 1);     display.println(" C");
    display.print("Humidity: "); display.print(humidity, 1); display.println(" %");
    display.print("Light:    "); display.println(lightRaw);
    display.print("Pressure: "); display.print(pressure, 0); display.println(" Pa");

    // Prediction + weather (if available)
    if (hasPrediction) {
        display.println();
        display.print(">> "); display.println(lastWeatherName);
        display.print("Sev: "); display.println(lastSeverity);
    } else {
        display.println();
        display.print("Buf: "); display.print(bufferCount);
        display.print("/"); display.println(NN_LOOKBACK);
    }

    display.display();

    // ── Stay awake so OLED + BLE are visible ─────────────────
    delay(WAKE_SECONDS * 1000);

    // ── Shutdown ─────────────────────────────────────────────
    BLEDevice::deinit(true);  // clean up BLE before sleep
    digitalWrite(SENSOR_POWER_PIN, LOW);

    Serial.println("Going to sleep...");
    esp_sleep_enable_timer_wakeup((uint64_t)SLEEP_SECONDS * 1000000ULL);
    esp_deep_sleep_start();
}

void loop() {
    // Empty — deep sleep always reboots into setup()
}
