import pandas as pd

from ftis.features import engineer_turbulence_features


def test_feature_engineering_creates_fti_and_labels() -> None:
    frame = pd.DataFrame(
        [
            {
                "latitude": 39.0,
                "longitude": -104.0,
                "altitude": 10000,
                "windspeed": 42,
                "pressure": 1002,
                "temperature": -40,
            }
        ]
    )

    features = engineer_turbulence_features(frame)

    assert "FTI" in features
    assert features["turbulence_label"].iloc[0] in {"Low", "Moderate", "High"}
