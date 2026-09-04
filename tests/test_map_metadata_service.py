from src.data.map_metadata_service import MapMetadataService


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_map_metadata_is_clipped_to_operational_bbox(monkeypatch):
    payload = {
        "elements": [
            {
                "type": "way",
                "id": 1,
                "tags": {"building": "yes", "name": "poste"},
                "geometry": [
                    {"lat": 43.40, "lon": 5.30},
                    {"lat": 43.40, "lon": 5.50},
                    {"lat": 43.60, "lon": 5.50},
                    {"lat": 43.60, "lon": 5.30},
                    {"lat": 43.40, "lon": 5.30},
                ],
            },
            {
                "type": "way",
                "id": 2,
                "tags": {"highway": "primary"},
                "geometry": [
                    {"lat": 43.525, "lon": 5.30},
                    {"lat": 43.525, "lon": 5.50},
                ],
            },
            {"type": "node", "id": 3, "lat": 43.525, "lon": 5.442, "tags": {"amenity": "fire_station"}},
        ]
    }
    monkeypatch.setattr(
        "src.data.map_metadata_service.requests.post",
        lambda *args, **kwargs: _Response(payload),
    )

    result = MapMetadataService(cache_ttl_sec=30.0).fetch(43.525, 5.442, 6400.0)
    assert result["is_live_api"] is True
    assert result["feature_count"] == 3
    west, south, east, north = result["bbox"]
    assert result["categories"]["buildings"] == 1
    for feature in result["features"]:
        geometry = feature["geometry"]
        coordinates = geometry["coordinates"]
        if geometry["type"] == "Point":
            coordinates = [coordinates]
        elif geometry["type"] == "Polygon":
            coordinates = coordinates[0]
        elif geometry["type"] == "MultiLineString":
            coordinates = [point for line in coordinates for point in line]
        for lon, lat in coordinates:
            assert west - 1e-9 <= lon <= east + 1e-9
            assert south - 1e-9 <= lat <= north + 1e-9
