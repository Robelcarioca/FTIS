from dashboard.app import normalize_backend_url, prediction_endpoint, system_status


class FakeResponse:
    ok = True
    status_code = 200

    def json(self) -> dict[str, object]:
        return {
            "status": "healthy",
            "version": "test",
            "model": {"artifact_version": "unit"},
        }


def test_backend_url_normalization_accepts_base_url() -> None:
    assert normalize_backend_url("https://ftis.example.com/") == "https://ftis.example.com"


def test_backend_url_normalization_accepts_predict_url() -> None:
    assert normalize_backend_url("http://localhost:8000/predict") == "http://localhost:8000"


def test_prediction_endpoint_is_derived_from_backend_url() -> None:
    assert prediction_endpoint("https://ftis.example.com") == "https://ftis.example.com/predict"


def test_system_status_online(monkeypatch) -> None:
    monkeypatch.setattr("dashboard.app.requests.get", lambda *args, **kwargs: FakeResponse())

    status = system_status("https://ftis.example.com")

    assert status["status"] == "ONLINE"
    assert status["model_version"] == "unit"
