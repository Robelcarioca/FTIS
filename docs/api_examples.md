# FTIS API Examples

Start the service from the `FTIS` directory:

```bash
uvicorn api.main:app --reload
```

Single prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"latitude":39.8729,"longitude":-104.6737,"altitude":10800,"windspeed":42,"pressure":1002,"temperature":-42}'
```

Batch prediction:

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"records":[{"latitude":39.8729,"longitude":-104.6737,"altitude":10800,"windspeed":42,"pressure":1002,"temperature":-42},{"latitude":40.64,"longitude":-73.78,"altitude":9000,"windspeed":25,"pressure":1009,"temperature":-31}]}'
```

Route analysis without live provider calls:

```bash
curl -X POST http://localhost:8000/route/analyze \
  -H "Content-Type: application/json" \
  -d '{"departure_airport":"LAX","destination_airport":"JFK","cruising_altitude_m":11000,"aircraft_speed_kt":455,"waypoint_count":32,"use_live_weather":false}'
```

Live weather:

```bash
curl -X POST http://localhost:8000/weather/live \
  -H "Content-Type: application/json" \
  -d '{"latitude":39.8561,"longitude":-104.6737,"altitude_m":10600,"station_id":"KDEN"}'
```

System status:

```bash
curl http://localhost:8000/system/status
```
