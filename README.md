# Streamable TTS API

## How to vLLM

1. run vLLM in background,

```bash
docker compose -f vllm.yaml up --detach
```

You can use direct vLLM API access for `TTS_API`, set `TTS_API=http://tts-engine:9090` in [.env](.env).

## API

1. Add `.env`, follow [.env_example](.env_example)

You can add more variables based on [app/main.py](app/main.py).

### Default run in GPU

```bash
docker compose up --build
```

### Default run in CPU

```bash
docker compose -f docker-compose-cpu.yaml up --build
```

### Streaming to file

```bash
curl -X POST \
  'http://localhost:9091/v1/audio/speech' \
  -H 'accept: audio/wav' \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Hello! How can I help you? can I get your passport number sir.",
    "voice": "husein",
    "model": "Scicom-intl/Multilingual-TTS-1.7B-v0.1",
    "response_format": "wav",
    "temperature": 0.7,
    "stream": true,
    "playback_speed": 0.5,
    "playback_overlap_speed": 0.1,
    "normalize": true
  }' \
  --output output.wav
```

## Stress test

This will stress test current worker size based on [docker-compose.yaml](docker-compose.yaml), to stress test is simple,

```bash
docker compose -f stress-test.yaml up --build
```

This will use default parameter from [app/env.py](app.py).

### H100 NVL

This one based on H100 NVL in Azure Standard_NC40ads_H100_v5

#### dynamic batching v1, older private commit

```
stress-test  | === SLO LATENCY REPORT ===
stress-test  | Total Requests: 50
stress-test  | Min Time: 1.606s
stress-test  | Max Time: 3.539s
stress-test  | Avg Time: 2.676s
stress-test  | P50 (Median): 2.568s
stress-test  | P90: 3.431s
stress-test  | P95: 3.474s
stress-test  | P99: 3.510s
stress-test  | 
stress-test  | === AUDIO & REAL-TIME FACTOR (RTF) REPORT ===
stress-test  | Avg Audio Duration: 5.702s
stress-test  | 
stress-test  | --- RTF Percentiles ---
stress-test  | Min RTF: 0.294s
stress-test  | Max RTF: 0.591s
stress-test  | Avg RTF: 0.470
stress-test  | P50 RTF: 0.434
stress-test  | P90 RTF: 0.581
stress-test  | P95 RTF: 0.587
stress-test  | P99 RTF: 0.589
stress-test  | ==========================
```

#### dynamic batching v2

```
stress-test  | === SLO LATENCY REPORT ===
stress-test  | Total Requests: 50
stress-test  | Min Time: 1.840s
stress-test  | Max Time: 2.296s
stress-test  | Avg Time: 2.164s
stress-test  | P50 (Median): 2.198s
stress-test  | P90: 2.286s
stress-test  | P95: 2.290s
stress-test  | P99: 2.295s
stress-test  | 
stress-test  | === AUDIO & REAL-TIME FACTOR (RTF) REPORT ===
stress-test  | Avg Audio Duration: 5.573s
stress-test  | 
stress-test  | --- RTF Percentiles ---
stress-test  | Min RTF: 0.329s
stress-test  | Max RTF: 0.554s
stress-test  | Avg RTF: 0.392
stress-test  | P50 RTF: 0.379
stress-test  | P90 RTF: 0.427
stress-test  | P95 RTF: 0.503
stress-test  | P99 RTF: 0.529
stress-test  | ==========================
```

### GPU RTX 3090 Ti

#### dynamic batching

Use [.env_dynamicbatching](.env_dynamicbatching).

```
stress-test  | === SLO LATENCY REPORT ===
stress-test  | Total Requests: 50
stress-test  | Min Time: 6.229s
stress-test  | Max Time: 6.545s
stress-test  | Avg Time: 6.429s
stress-test  | P50 (Median): 6.424s
stress-test  | P90: 6.530s
stress-test  | P95: 6.543s
stress-test  | P99: 6.544s
stress-test  | 
stress-test  | === AUDIO & REAL-TIME FACTOR (RTF) REPORT ===
stress-test  | Avg Audio Duration: 5.759s
stress-test  | 
stress-test  | --- RTF Percentiles ---
stress-test  | Min RTF: 0.966s
stress-test  | Max RTF: 1.702s
stress-test  | Avg RTF: 1.124
stress-test  | P50 RTF: 1.099
stress-test  | P90 RTF: 1.194
stress-test  | P95 RTF: 1.203
stress-test  | P99 RTF: 1.405
stress-test  | ==========================
```

#### dynamic batching with CUDA Graphs

Use [.env_dynamicbatching_cudagraph](.env_dynamicbatching_cudagraph).

```
stress-test  | === SLO LATENCY REPORT ===
stress-test  | Total Requests: 50
stress-test  | Min Time: 6.125s
stress-test  | Max Time: 6.323s
stress-test  | Avg Time: 6.262s
stress-test  | P50 (Median): 6.273s
stress-test  | P90: 6.313s
stress-test  | P95: 6.322s
stress-test  | P99: 6.322s
stress-test  | 
stress-test  | === AUDIO & REAL-TIME FACTOR (RTF) REPORT ===
stress-test  | Avg Audio Duration: 5.731s
stress-test  | 
stress-test  | --- RTF Percentiles ---
stress-test  | Min RTF: 0.983s
stress-test  | Max RTF: 1.701s
stress-test  | Avg RTF: 1.105
stress-test  | P50 RTF: 1.067
stress-test  | P90 RTF: 1.160
stress-test  | P95 RTF: 1.278
stress-test  | P99 RTF: 1.540
stress-test  | ==========================
```

#### Without dynamic batching

Use [.env_dynamicbatching](.env_nodynamicbatching).

```
stress-test  | === SLO LATENCY REPORT ===
stress-test  | Total Requests: 50
stress-test  | Min Time: 4.847s
stress-test  | Max Time: 7.734s
stress-test  | Avg Time: 7.136s
stress-test  | P50 (Median): 7.291s
stress-test  | P90: 7.637s
stress-test  | P95: 7.667s
stress-test  | P99: 7.710s
stress-test  | 
stress-test  | === AUDIO & REAL-TIME FACTOR (RTF) REPORT ===
stress-test  | Avg Audio Duration: 5.630s
stress-test  | 
stress-test  | --- RTF Percentiles ---
stress-test  | Min RTF: 1.228s
stress-test  | Max RTF: 1.282s
stress-test  | Avg RTF: 1.267
stress-test  | P50 RTF: 1.269
stress-test  | P90 RTF: 1.278
stress-test  | P95 RTF: 1.281
stress-test  | P99 RTF: 1.281
stress-test  | ==========================
```