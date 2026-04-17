import { useState, useEffect, useRef, useCallback } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend,
} from "recharts";

const SERVICE_UUID = "e3a1f0b0-1234-5678-abcd-000000000001";
const CHAR_CURRENT = "e3a1f0b0-1234-5678-abcd-000000000002";
const CHAR_PREDICTED = "e3a1f0b0-1234-5678-abcd-000000000003";
const CHAR_WEATHER = "e3a1f0b0-1234-5678-abcd-000000000004";
const CHAR_ALERT = "e3a1f0b0-1234-5678-abcd-000000000005";
const CHAR_TREND = "e3a1f0b0-1234-5678-abcd-000000000006";
const CHAR_HISTORY = "e3a1f0b0-1234-5678-abcd-000000000007";

const mono = "'JetBrains Mono', monospace";
const displayFont = "'Space Mono', monospace";
const body = "'DM Sans', sans-serif";
const card = {
  background: "rgba(255,255,255,0.025)",
  border: "1px solid rgba(255,255,255,0.06)",
  borderRadius: 20,
  backdropFilter: "blur(12px)",
  position: "relative",
  zIndex: 2,
};

const WEATHER_COLORS = {
  CLEAR: "#66bb6a",
  CLOUDY: "#90a4ae",
  OVERCAST: "#78909c",
  FOG: "#b0bec5",
  RAIN: "#42a5f5",
  SNOW: "#e0e0e0",
  THUNDERSTORM: "#ffa726",
  "STORM WATCH": "#ff7043",
  "HEAT WAVE": "#ef5350",
  HURRICANE: "#e53935",
  BLIZZARD: "#80deea",
  "NO DATA": "#6a6a7a",
  UNKNOWN: "#6a6a7a",
  WAITING: "#6a6a7a",
};

const CLOUD_CHARS = ["\u2601", "\u26C5", "\u2600", "\u26C8", "\u2744"];

function FloatingCloud({
  char,
  delay,
  duration,
  startX,
  vertical,
  swayAmount,
  size,
  opacity,
}) {
  const id = `drift_${delay.toString().replace(".", "_")}_${startX.toString().replace(".", "_")}`;
  const midSwayY = -30 + Math.random() * 60;
  const endSwayY = -15 + Math.random() * 30;
  const midRot = 20 + Math.random() * 40;
  const endRot = 40 + Math.random() * 60;
  const kf = `@keyframes ${id}{0%{transform:translate(0,0) rotate(0deg);opacity:0}8%{opacity:${opacity}}35%{transform:translate(${swayAmount * 0.4}px,${midSwayY}px) rotate(${midRot}deg)}65%{transform:translate(${swayAmount * 0.75}px,${midSwayY * -0.6}px) rotate(${midRot + 20}deg)}92%{opacity:${opacity * 0.6}}100%{transform:translate(${swayAmount}px,${endSwayY}px) rotate(${endRot}deg);opacity:0}}`;
  return (
    <>
      <style>{kf}</style>
      <span
        style={{
          position: "absolute",
          left: `${startX}%`,
          top: `${vertical}%`,
          fontSize: size,
          animation: `${id} ${duration}s ease-in-out ${delay}s infinite`,
          pointerEvents: "none",
          opacity: 0,
          zIndex: 1,
        }}
      >
        {char}
      </span>
    </>
  );
}

function BurstCloud({ char, x, y, angle, distance, id }) {
  const rad = (angle * Math.PI) / 180;
  const tx = Math.cos(rad) * distance;
  const ty = Math.sin(rad) * distance;
  const rot = 60 + Math.random() * 120;
  const kf = `@keyframes burst_${id}{0%{transform:translate(0,0) rotate(0deg) scale(1);opacity:1}60%{opacity:0.8}100%{transform:translate(${tx}px,${ty}px) rotate(${rot}deg) scale(0.3);opacity:0}}`;
  return (
    <>
      <style>{kf}</style>
      <span
        style={{
          position: "fixed",
          left: x,
          top: y,
          fontSize: 16 + Math.random() * 10,
          animation: `burst_${id} 1s cubic-bezier(.2,.8,.3,1) forwards`,
          pointerEvents: "none",
          zIndex: 9999,
        }}
      >
        {char}
      </span>
    </>
  );
}

function Gauge({ label, value, unit, min, max, color, icon }) {
  const pct =
    value !== null
      ? Math.min(100, Math.max(0, ((value - min) / (max - min)) * 100))
      : 0;
  const radius = 58,
    circ = 2 * Math.PI * radius,
    arc = circ * 0.75;
  const offset = arc - (pct / 100) * arc;
  return (
    <div
      style={{
        ...card,
        padding: "24px 16px 16px",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
      }}
    >
      <span
        style={{
          fontSize: 10,
          letterSpacing: 2.5,
          color: "#7a7a8a",
          textTransform: "uppercase",
          fontFamily: mono,
          marginBottom: 8,
        }}
      >
        {icon} {label}
      </span>
      <svg width="140" height="115" viewBox="0 0 140 130">
        <circle
          cx="70"
          cy="75"
          r={radius}
          fill="none"
          stroke="rgba(255,255,255,0.04)"
          strokeWidth="9"
          strokeDasharray={`${arc} ${circ}`}
          strokeLinecap="round"
          transform="rotate(135 70 75)"
        />
        <circle
          cx="70"
          cy="75"
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth="9"
          strokeDasharray={`${arc} ${circ}`}
          strokeDashoffset={offset}
          strokeLinecap="round"
          transform="rotate(135 70 75)"
          style={{
            transition: "stroke-dashoffset 0.8s cubic-bezier(.4,0,.2,1)",
          }}
        />
        <text
          x="70"
          y="70"
          textAnchor="middle"
          fill="#e8e8f0"
          fontSize="26"
          fontFamily={displayFont}
          fontWeight="700"
        >
          {value !== null ? value : "\u2014"}
        </text>
        <text
          x="70"
          y="93"
          textAnchor="middle"
          fill="#5a5a6a"
          fontSize="12"
          fontFamily={mono}
        >
          {unit}
        </text>
      </svg>
    </div>
  );
}

function StatusPill({ label, value, ok }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 10,
        padding: "10px 16px",
        background: "rgba(255,255,255,0.02)",
        borderRadius: 12,
        border: "1px solid rgba(255,255,255,0.04)",
        position: "relative",
        zIndex: 2,
      }}
    >
      <div
        style={{
          width: 8,
          height: 8,
          borderRadius: "50%",
          background: ok ? "#66bb6a" : "#ef5350",
          boxShadow: ok
            ? "0 0 8px rgba(102,187,106,0.4)"
            : "0 0 8px rgba(239,83,80,0.4)",
        }}
      />
      <span
        style={{ fontSize: 12, color: "#7a7a8a", fontFamily: mono, flex: 1 }}
      >
        {label}
      </span>
      <span style={{ fontSize: 12, color: "#b0b0bc", fontFamily: mono }}>
        {value}
      </span>
    </div>
  );
}

function WeatherBadge({ name, severity }) {
  const c = WEATHER_COLORS[name] || "#6a6a7a";
  const sevLabels = ["CLEAR", "MILD", "MODERATE", "SEVERE"];
  return (
    <div
      style={{
        ...card,
        padding: "24px 16px 16px",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <span
        style={{
          fontSize: 10,
          letterSpacing: 2.5,
          color: "#7a7a8a",
          textTransform: "uppercase",
          fontFamily: mono,
        }}
      >
        Forecast
      </span>
      <span
        style={{
          fontSize: 28,
          fontWeight: 700,
          fontFamily: displayFont,
          marginTop: 10,
          color: c,
          transition: "color 0.5s",
          textAlign: "center",
          lineHeight: 1.2,
        }}
      >
        {name || "\u2014"}
      </span>
      <span
        style={{
          fontSize: 11,
          color: c,
          fontFamily: mono,
          marginTop: 4,
          transition: "color 0.5s",
          opacity: 0.7,
        }}
      >
        {sevLabels[severity] || ""}
      </span>
      {severity >= 2 && (
        <div
          style={{
            marginTop: 8,
            padding: "3px 10px",
            borderRadius: 6,
            background: "rgba(239,83,80,0.15)",
            border: "1px solid rgba(239,83,80,0.3)",
          }}
        >
          <span
            style={{
              fontSize: 10,
              color: "#ef5350",
              fontFamily: mono,
              letterSpacing: 1,
            }}
          >
            ALERT
          </span>
        </div>
      )}
    </div>
  );
}

export default function WeatherMindDashboard() {
  const [connected, setConnected] = useState(false);
  const [bleError, setBleError] = useState("");
  const [readingCount, setReadingCount] = useState(0);
  const [startTime] = useState(Date.now());
  const [burstClouds, setBurstClouds] = useState([]);
  const burstIdRef = useRef(0);

  const [current, setCurrent] = useState({
    temp: null,
    hum: null,
    pres: null,
    lux: null,
  });
  const [predicted, setPredicted] = useState({
    temp: null,
    hum: null,
    pres: null,
  });
  const [weather, setWeather] = useState({
    name: "WAITING",
    severity: 0,
    detail: "Waiting for data",
  });
  const [alertFlag, setAlertFlag] = useState(false);
  const [trend, setTrend] = useState({ label: "--", slope: 0 });
  const [history, setHistory] = useState([]);

  useEffect(() => {
    if (burstClouds.length > 0) {
      const t = setTimeout(() => setBurstClouds([]), 1200);
      return () => clearTimeout(t);
    }
  }, [burstClouds]);

  const spawnBurst = (e) => {
    const clouds = [];
    for (let i = 0; i < 6; i++) {
      clouds.push({
        id: burstIdRef.current++,
        char: CLOUD_CHARS[Math.floor(Math.random() * CLOUD_CHARS.length)],
        x: e.clientX,
        y: e.clientY,
        angle: Math.random() * 360,
        distance: 40 + Math.random() * 80,
      });
    }
    setBurstClouds(clouds);
  };

  const parseCSV = (str) => {
    const p = str.split(",").map((s) => parseFloat(s.trim()));
    return p;
  };

  const connectBLE = useCallback(async () => {
    setBleError("");
    try {
      const device = await navigator.bluetooth.requestDevice({
        filters: [{ namePrefix: "WeatherMind" }],
        optionalServices: [SERVICE_UUID],
      });
      const server = await device.gatt.connect();
      const service = await server.getPrimaryService(SERVICE_UUID);

      const readChar = async (uuid) => {
        try {
          const ch = await service.getCharacteristic(uuid);
          const val = await ch.readValue();
          return new TextDecoder().decode(val);
        } catch {
          return null;
        }
      };

      const refresh = async () => {
        const curStr = await readChar(CHAR_CURRENT);
        if (curStr) {
          const p = parseCSV(curStr);
          if (p.length >= 3) {
            setCurrent({
              temp: p[0],
              hum: p[1],
              pres: p[2],
              lux: p[3] || null,
            });
            setHistory((prev) => [
              ...prev.slice(-299),
              {
                temp: p[0],
                hum: p[1],
                pres: p[2],
                lux: p[3] || 0,
                ts: Date.now(),
              },
            ]);
            setReadingCount((prev) => prev + 1);
          }
        }

        const predStr = await readChar(CHAR_PREDICTED);
        if (predStr) {
          const p = parseCSV(predStr);
          if (p.length >= 3)
            setPredicted({ temp: p[0], hum: p[1], pres: p[2] });
        }

        const wStr = await readChar(CHAR_WEATHER);
        if (wStr) {
          const parts = wStr.split("|");
          setWeather({
            name: parts[0] || "UNKNOWN",
            severity: parseInt(parts[1]) || 0,
            detail: parts[2] || "",
          });
        }

        const aStr = await readChar(CHAR_ALERT);
        if (aStr) setAlertFlag(aStr.trim() === "1");

        const tStr = await readChar(CHAR_TREND);
        if (tStr) {
          const parts = tStr.split("|");
          setTrend({
            label: parts[0] || "--",
            slope: parseFloat(parts[1]) || 0,
          });
        }
      };

      await refresh();

      const interval = setInterval(async () => {
        try {
          if (device.gatt.connected) await refresh();
          else {
            clearInterval(interval);
            setConnected(false);
          }
        } catch {
          clearInterval(interval);
          setConnected(false);
        }
      }, 5000);

      device.addEventListener("gattserverdisconnected", () => {
        clearInterval(interval);
        setConnected(false);
      });

      setConnected(true);
    } catch (err) {
      setBleError(err.message || "Connection failed");
    }
  }, []);

  const loadDemoData = () => {
    const t = +(20 + Math.random() * 15).toFixed(1);
    const h = +(40 + Math.random() * 40).toFixed(1);
    const p = +(98000 + Math.random() * 4000).toFixed(0);
    const l = +(Math.random() * 632).toFixed(0);
    setCurrent({ temp: t, hum: h, pres: +p, lux: +l });
    setPredicted({
      temp: +(t + (-2 + Math.random() * 4)).toFixed(1),
      hum: +(h + (-5 + Math.random() * 10)).toFixed(1),
      pres: +(p + (-200 + Math.random() * 400)).toFixed(0),
    });
    const weathers = [
      "CLEAR",
      "CLOUDY",
      "OVERCAST",
      "RAIN",
      "THUNDERSTORM",
      "FOG",
    ];
    const wn = weathers[Math.floor(Math.random() * weathers.length)];
    const sv =
      wn === "CLEAR"
        ? 0
        : wn === "CLOUDY" || wn === "FOG"
          ? 1
          : wn === "RAIN" || wn === "OVERCAST"
            ? 2
            : 3;
    setWeather({ name: wn, severity: sv, detail: "Demo data" });
    setAlertFlag(sv >= 2);
    setTrend({
      label: ["STEADY", "RISING", "FALLING"][Math.floor(Math.random() * 3)],
      slope: +(-2 + Math.random() * 4).toFixed(2),
    });
    setHistory((prev) => [
      ...prev.slice(-299),
      { temp: t, hum: h, pres: +p, lux: +l, ts: Date.now() },
    ]);
    setReadingCount((prev) => prev + 1);
  };

  const uptime = Math.floor((Date.now() - startTime) / 60000);
  const wColor = WEATHER_COLORS[weather.name] || "#6a6a7a";

  const chartData = history.slice(-30).map((r) => ({
    time: new Date(r.ts).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    }),
    Temp: r.temp,
    Humidity: r.hum,
    "P (mb)": +(r.pres / 100).toFixed(1),
  }));

  const ambientClouds = useRef(
    Array.from({ length: 60 }, (_, i) => ({
      char: CLOUD_CHARS[i % CLOUD_CHARS.length],
      delay: i * 1.2 + Math.random() * 3,
      duration: 22 + Math.random() * 16,
      startX: -8 + Math.random() * 5,
      vertical: Math.random() * 95,
      swayAmount:
        (typeof window !== "undefined" ? window.innerWidth : 1200) +
        Math.random() * 200,
      size: 10 + Math.random() * 14,
      opacity: 0.06 + Math.random() * 0.08,
    })),
  ).current;

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#08080d",
        color: "#e0e0ec",
        fontFamily: body,
        display: "flex",
        flexDirection: "column",
        position: "relative",
        overflow: "hidden",
      }}
    >
      <link
        href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Mono:wght@400;700&family=JetBrains+Mono:wght@400;500&display=swap"
        rel="stylesheet"
      />

      <div
        style={{
          position: "fixed",
          inset: 0,
          pointerEvents: "none",
          zIndex: 1,
          overflow: "hidden",
        }}
      >
        {ambientClouds.map((c, i) => (
          <FloatingCloud key={i} {...c} />
        ))}
      </div>

      {burstClouds.map((c) => (
        <BurstCloud key={c.id} {...c} />
      ))}

      <header
        style={{
          padding: "28px 40px 22px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          borderBottom: "1px solid rgba(255,255,255,0.04)",
          flexShrink: 0,
          position: "relative",
          zIndex: 2,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div
            style={{
              width: 42,
              height: 42,
              borderRadius: 12,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              background:
                "linear-gradient(135deg, rgba(66,165,245,0.15), rgba(255,167,38,0.15))",
              border: "1px solid rgba(66,165,245,0.2)",
              fontSize: 20,
            }}
          >
            {"\u26C5"}
          </div>
          <div>
            <h1
              style={{
                margin: 0,
                fontSize: 22,
                fontWeight: 700,
                letterSpacing: -0.5,
                fontFamily: displayFont,
                color: "#f0f0f8",
              }}
            >
              WeatherMind
            </h1>
            <p
              style={{
                margin: "2px 0 0",
                fontSize: 11,
                color: "#5a5a6a",
                letterSpacing: 1.5,
                textTransform: "uppercase",
                fontFamily: mono,
              }}
            >
              by Shadow Mechanics
            </p>
          </div>
        </div>
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <button
            onClick={(e) => {
              loadDemoData();
              spawnBurst(e);
            }}
            style={{
              padding: "8px 16px",
              borderRadius: 10,
              border: "1px solid rgba(255,255,255,0.07)",
              background: "rgba(255,255,255,0.02)",
              color: "#6a6a7a",
              cursor: "pointer",
              fontSize: 11,
              fontFamily: mono,
            }}
          >
            + Demo Data
          </button>
          <button
            onClick={(e) => {
              connectBLE();
              spawnBurst(e);
            }}
            style={{
              padding: "8px 20px",
              borderRadius: 10,
              border: connected
                ? "1px solid rgba(76,175,80,0.25)"
                : "1px solid rgba(100,180,255,0.25)",
              background: connected
                ? "rgba(76,175,80,0.08)"
                : "rgba(100,180,255,0.08)",
              color: connected ? "#81c784" : "#90caf9",
              cursor: "pointer",
              fontSize: 11,
              fontFamily: mono,
            }}
          >
            {connected ? "\u25CF Connected" : "Connect BLE"}
          </button>
        </div>
      </header>
      {bleError && (
        <p
          style={{
            color: "#ef5350",
            fontSize: 11,
            padding: "0 40px",
            margin: "8px 0 0",
            fontFamily: mono,
            position: "relative",
            zIndex: 2,
          }}
        >
          Error: {bleError}
        </p>
      )}

      <main
        style={{ flex: 1, padding: "0 40px", position: "relative", zIndex: 2 }}
      >
        <section
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(5, 1fr)",
            gap: 14,
            marginTop: 24,
          }}
        >
          <Gauge
            label="Temperature"
            value={current.temp}
            unit="\u00B0C"
            min={-10}
            max={50}
            color="#ef5350"
            icon={"\uD83C\uDF21"}
          />
          <Gauge
            label="Humidity"
            value={current.hum}
            unit="%"
            min={0}
            max={100}
            color="#42a5f5"
            icon={"\uD83D\uDCA7"}
          />
          <Gauge
            label="Pressure"
            value={current.pres ? +(current.pres / 100).toFixed(1) : null}
            unit="mb"
            min={960}
            max={1050}
            color="#ab47bc"
            icon={"\u2B07"}
          />
          <Gauge
            label="Light"
            value={current.lux}
            unit="lux"
            min={0}
            max={632}
            color="#ffa726"
            icon={"\u2600"}
          />
          <WeatherBadge name={weather.name} severity={weather.severity} />
        </section>

        <section
          style={{
            display: "grid",
            gridTemplateColumns: "2fr 1fr",
            gap: 14,
            marginTop: 14,
          }}
        >
          <div style={{ ...card, padding: 24 }}>
            <h3
              style={{
                margin: "0 0 16px",
                fontSize: 11,
                letterSpacing: 2.5,
                color: "#6a6a7a",
                textTransform: "uppercase",
                fontFamily: mono,
              }}
            >
              Sensor Trends
            </h3>
            {chartData.length > 1 ? (
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={chartData}>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="rgba(255,255,255,0.04)"
                  />
                  <XAxis
                    dataKey="time"
                    tick={{ fill: "#5a5a6a", fontSize: 9 }}
                    axisLine={{ stroke: "rgba(255,255,255,0.06)" }}
                  />
                  <YAxis
                    tick={{ fill: "#5a5a6a", fontSize: 9 }}
                    axisLine={{ stroke: "rgba(255,255,255,0.06)" }}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "#13131c",
                      border: "1px solid rgba(255,255,255,0.08)",
                      borderRadius: 10,
                      fontSize: 11,
                      fontFamily: mono,
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: 10, fontFamily: mono }} />
                  <Line
                    type="monotone"
                    dataKey="Temp"
                    stroke="#ef5350"
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="Humidity"
                    stroke="#42a5f5"
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="P (mb)"
                    stroke="#ab47bc"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div
                style={{
                  height: 220,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  color: "#3a3a4a",
                  fontSize: 13,
                  fontFamily: mono,
                }}
              >
                Waiting for sensor data...
              </div>
            )}
          </div>

          <div style={{ ...card, padding: 24 }}>
            <h3
              style={{
                margin: "0 0 18px",
                fontSize: 11,
                letterSpacing: 2.5,
                color: "#6a6a7a",
                textTransform: "uppercase",
                fontFamily: mono,
              }}
            >
              30-Min Forecast
            </h3>
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              {[
                {
                  label: "Temperature",
                  cur: current.temp,
                  pred: predicted.temp,
                  unit: "\u00B0C",
                  color: "#ef5350",
                },
                {
                  label: "Humidity",
                  cur: current.hum,
                  pred: predicted.hum,
                  unit: "%",
                  color: "#42a5f5",
                },
                {
                  label: "Pressure",
                  cur: current.pres ? +(current.pres / 100).toFixed(1) : null,
                  pred: predicted.pres
                    ? +(predicted.pres / 100).toFixed(1)
                    : null,
                  unit: "mb",
                  color: "#ab47bc",
                },
              ].map((r) => {
                const delta =
                  r.cur !== null && r.pred !== null ? r.pred - r.cur : null;
                const arrow =
                  delta === null
                    ? ""
                    : delta > 0.1
                      ? "\u2191"
                      : delta < -0.1
                        ? "\u2193"
                        : "\u2192";
                const deltaColor =
                  delta === null
                    ? "#5a5a6a"
                    : delta > 0.5
                      ? "#ef5350"
                      : delta < -0.5
                        ? "#42a5f5"
                        : "#66bb6a";
                return (
                  <div key={r.label}>
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        marginBottom: 6,
                      }}
                    >
                      <span
                        style={{
                          fontSize: 11,
                          color: "#8a8a9a",
                          fontFamily: mono,
                        }}
                      >
                        {r.label}
                      </span>
                      <span
                        style={{
                          fontSize: 11,
                          color: deltaColor,
                          fontFamily: mono,
                        }}
                      >
                        {r.pred !== null
                          ? `${r.pred}${r.unit} ${arrow}`
                          : "\u2014"}
                      </span>
                    </div>
                    <div
                      style={{
                        position: "relative",
                        height: 6,
                        background: "rgba(255,255,255,0.04)",
                        borderRadius: 3,
                      }}
                    >
                      <div
                        style={{
                          position: "absolute",
                          left: 0,
                          height: "100%",
                          width: r.pred !== null ? "100%" : "0%",
                          background: `${r.color}15`,
                          borderRadius: 3,
                          transition: "width 0.6s",
                        }}
                      />
                    </div>
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        marginTop: 3,
                      }}
                    >
                      <span
                        style={{
                          fontSize: 9,
                          color: "#4a4a5a",
                          fontFamily: mono,
                        }}
                      >
                        Now: {r.cur !== null ? `${r.cur}${r.unit}` : "\u2014"}
                      </span>
                      <span
                        style={{
                          fontSize: 9,
                          color: "#4a4a5a",
                          fontFamily: mono,
                        }}
                      >
                        Trend: {trend.label}
                      </span>
                    </div>
                  </div>
                );
              })}

              <div
                style={{
                  marginTop: 8,
                  padding: "12px 16px",
                  background: "rgba(255,255,255,0.02)",
                  borderRadius: 12,
                  border: "1px solid rgba(255,255,255,0.04)",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                  }}
                >
                  <span
                    style={{ fontSize: 11, color: "#7a7a8a", fontFamily: mono }}
                  >
                    Classification
                  </span>
                  <span
                    style={{
                      fontSize: 13,
                      color: wColor,
                      fontFamily: mono,
                      fontWeight: 700,
                    }}
                  >
                    {weather.name}
                  </span>
                </div>
                {weather.detail && (
                  <p
                    style={{
                      margin: "6px 0 0",
                      fontSize: 10,
                      color: "#5a5a6a",
                      fontFamily: mono,
                    }}
                  >
                    {weather.detail}
                  </p>
                )}
              </div>
            </div>
          </div>
        </section>

        <section
          style={{
            display: "grid",
            gridTemplateColumns: "1fr",
            gap: 14,
            marginTop: 14,
          }}
        >
          <div
            style={{ ...card, padding: 24, maxHeight: 350, overflowY: "auto" }}
          >
            <h3
              style={{
                margin: "0 0 14px",
                fontSize: 11,
                letterSpacing: 2.5,
                color: "#6a6a7a",
                textTransform: "uppercase",
                fontFamily: mono,
              }}
            >
              Reading History
            </h3>
            {history.length === 0 ? (
              <p style={{ color: "#3a3a4a", fontSize: 12, fontFamily: mono }}>
                No readings yet. Connect BLE or load demo data.
              </p>
            ) : (
              <table
                style={{
                  width: "100%",
                  borderCollapse: "collapse",
                  fontSize: 11,
                  fontFamily: mono,
                }}
              >
                <thead>
                  <tr
                    style={{
                      color: "#5a5a6a",
                      borderBottom: "1px solid rgba(255,255,255,0.06)",
                    }}
                  >
                    <th
                      style={{
                        textAlign: "left",
                        padding: "6px 0",
                        fontWeight: 500,
                      }}
                    >
                      Time
                    </th>
                    <th style={{ fontWeight: 500 }}>Temp</th>
                    <th style={{ fontWeight: 500 }}>Hum</th>
                    <th style={{ fontWeight: 500 }}>Press (mb)</th>
                    <th style={{ fontWeight: 500 }}>Lux</th>
                  </tr>
                </thead>
                <tbody>
                  {[...history]
                    .reverse()
                    .slice(0, 20)
                    .map((r, i) => (
                      <tr
                        key={i}
                        style={{
                          borderBottom: "1px solid rgba(255,255,255,0.025)",
                          color: "#a0a0ac",
                        }}
                      >
                        <td style={{ padding: "5px 0" }}>
                          {new Date(r.ts).toLocaleTimeString()}
                        </td>
                        <td style={{ textAlign: "center" }}>{r.temp}</td>
                        <td style={{ textAlign: "center" }}>{r.hum}</td>
                        <td style={{ textAlign: "center" }}>
                          {(r.pres / 100).toFixed(1)}
                        </td>
                        <td style={{ textAlign: "center" }}>{r.lux}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            )}
          </div>
        </section>

        <section
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(4, 1fr)",
            gap: 14,
            marginTop: 14,
            marginBottom: 24,
          }}
        >
          <StatusPill
            label="BLE Connection"
            value={connected ? "Active" : "Disconnected"}
            ok={connected}
          />
          <StatusPill
            label="Total Readings"
            value={readingCount.toString()}
            ok={readingCount > 0}
          />
          <StatusPill label="Session Uptime" value={`${uptime}m`} ok={true} />
          <StatusPill
            label="Alert Status"
            value={alertFlag ? "ACTIVE" : "None"}
            ok={!alertFlag}
          />
        </section>
      </main>

      <footer
        style={{
          padding: "24px 40px 20px",
          borderTop: "1px solid rgba(255,255,255,0.04)",
          flexShrink: 0,
          position: "relative",
          zIndex: 2,
        }}
      >
        <div
          style={{
            position: "absolute",
            bottom: 0,
            left: 0,
            right: 0,
            height: 3,
          }}
        >
          <div
            style={{
              height: "100%",
              background:
                "linear-gradient(90deg, transparent 0%, rgba(66,165,245,0.08) 15%, rgba(171,71,188,0.12) 30%, transparent 45%, rgba(255,167,38,0.06) 60%, rgba(239,83,80,0.1) 75%, transparent 100%)",
            }}
          />
        </div>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <span style={{ fontSize: 12, color: "#4a4a5a", fontFamily: mono }}>
            WeatherMind v1.0
          </span>
          <span style={{ fontSize: 11, color: "#3a3a4a", fontFamily: mono }}>
            ESP32 + NN + BLE + Kalman
          </span>
        </div>
      </footer>
    </div>
  );
}
