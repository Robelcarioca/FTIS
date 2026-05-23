"""Small built-in airport coordinate registry for demos and tests."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Airport:
    code: str
    name: str
    latitude: float
    longitude: float


AIRPORTS: dict[str, Airport] = {
    "ADD": Airport("HAAB", "Addis Ababa Bole International", 8.9779, 38.7993),
    "DIR": Airport("HADR", "Dire Dawa Aba Tenna Dejazmach Yilma", 9.6247, 41.8542),
    "BJR": Airport("HABD", "Bahir Dar", 11.6081, 37.3216),
    "MQX": Airport("HAMK", "Mekelle Alula Aba Nega", 13.4674, 39.5335),
    "JIM": Airport("HAJM", "Jimma Aba Jifar", 7.6661, 36.8166),
    "GDQ": Airport("HAGN", "Gondar Atse Tewodros", 12.5199, 37.4339),
    "AWA": Airport("HALA", "Hawassa", 7.0670, 38.5000),
    "ATL": Airport("KATL", "Hartsfield-Jackson Atlanta", 33.6407, -84.4277),
    "BOS": Airport("KBOS", "Boston Logan", 42.3656, -71.0096),
    "DEN": Airport("KDEN", "Denver International", 39.8561, -104.6737),
    "DFW": Airport("KDFW", "Dallas/Fort Worth", 32.8998, -97.0403),
    "JFK": Airport("KJFK", "John F. Kennedy International", 40.6413, -73.7781),
    "LAS": Airport("KLAS", "Harry Reid International", 36.0840, -115.1537),
    "LAX": Airport("KLAX", "Los Angeles International", 33.9416, -118.4085),
    "MIA": Airport("KMIA", "Miami International", 25.7959, -80.2871),
    "ORD": Airport("KORD", "Chicago O'Hare", 41.9742, -87.9073),
    "PHX": Airport("KPHX", "Phoenix Sky Harbor", 33.4352, -112.0101),
    "SEA": Airport("KSEA", "Seattle-Tacoma", 47.4502, -122.3088),
    "SFO": Airport("KSFO", "San Francisco International", 37.6213, -122.3790),
    "YYZ": Airport("CYYZ", "Toronto Pearson", 43.6777, -79.6248),
    "LHR": Airport("EGLL", "London Heathrow", 51.4700, -0.4543),
    "CDG": Airport("LFPG", "Paris Charles de Gaulle", 49.0097, 2.5479),
    "DXB": Airport("OMDB", "Dubai International", 25.2532, 55.3657),
    "NBO": Airport("HKJK", "Jomo Kenyatta International", -1.3192, 36.9278),
    "JED": Airport("OEJN", "King Abdulaziz International", 21.6796, 39.1565),
    "FRA": Airport("EDDF", "Frankfurt Airport", 50.0379, 8.5622),
    "HKG": Airport("VHHH", "Hong Kong International", 22.3080, 113.9185),
    "HND": Airport("RJTT", "Tokyo Haneda", 35.5494, 139.7798),
    "SIN": Airport("WSSS", "Singapore Changi", 1.3644, 103.9915),
}

for airport in list(AIRPORTS.values()):
    AIRPORTS[airport.code] = airport


def resolve_airport(code: str) -> Airport:
    """Resolve IATA or ICAO code to airport metadata."""

    normalized = code.strip().upper()
    if normalized not in AIRPORTS:
        raise ValueError(f"Unknown airport code '{code}'. Add coordinates or use a known hub.")
    return AIRPORTS[normalized]
