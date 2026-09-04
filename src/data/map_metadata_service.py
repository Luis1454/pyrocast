"""OpenStreetMap metadata loader for the operational fire domain."""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests


Coordinate = Tuple[float, float]


class MapMetadataService:
    """Load and clip relevant OSM features to the active operational bbox."""

    OVERPASS_ENDPOINTS = (
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    )

    def __init__(self, cache_ttl_sec: float = 600.0, max_features: int = 2500):
        self.cache_ttl_sec = float(cache_ttl_sec)
        self.max_features = int(max_features)
        self._cache: Dict[Tuple[float, float, float], Tuple[float, Dict[str, Any]]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _bbox(center_lat: float, center_lon: float, domain_size_m: float) -> Tuple[float, float, float, float]:
        half = float(domain_size_m) / 2.0
        d_lat = half / 111320.0
        d_lon = half / max(0.2, 111320.0 * math.cos(math.radians(center_lat)))
        return center_lat - d_lat, center_lon - d_lon, center_lat + d_lat, center_lon + d_lon

    @staticmethod
    def _cache_key(center_lat: float, center_lon: float, domain_size_m: float) -> Tuple[float, float, float]:
        return round(float(center_lat), 4), round(float(center_lon), 4), round(float(domain_size_m), 1)

    @staticmethod
    def _category(tags: Dict[str, Any]) -> str:
        if tags.get("building"):
            return "buildings"
        if tags.get("highway"):
            return "roads"
        if tags.get("railway"):
            return "railways"
        if tags.get("waterway"):
            return "waterways"
        if tags.get("natural") in {"water", "wetland"} or tags.get("water"):
            return "water"
        if tags.get("landuse") or tags.get("natural") in {"wood", "forest", "scrub", "heath", "grassland", "meadow", "bare_rock"}:
            return "landuse"
        if tags.get("power"):
            return "power"
        if tags.get("barrier"):
            return "barriers"
        if tags.get("amenity") or tags.get("place") or tags.get("man_made"):
            return "points"
        return "other"

    @staticmethod
    def _clip_segment(a: Coordinate, b: Coordinate, bbox: Tuple[float, float, float, float]) -> Optional[List[Coordinate]]:
        south, west, north, east = bbox
        x0, y0 = a
        x1, y1 = b
        dx = x1 - x0
        dy = y1 - y0
        t0, t1 = 0.0, 1.0
        for p, q in ((-dx, x0 - west), (dx, east - x0), (-dy, y0 - south), (dy, north - y0)):
            if abs(p) < 1e-12:
                if q < 0.0:
                    return None
                continue
            ratio = q / p
            if p < 0.0:
                if ratio > t1:
                    return None
                t0 = max(t0, ratio)
            else:
                if ratio < t0:
                    return None
                t1 = min(t1, ratio)
        return [(x0 + t0 * dx, y0 + t0 * dy), (x0 + t1 * dx, y0 + t1 * dy)]

    @classmethod
    def _clip_line(cls, points: Sequence[Coordinate], bbox: Tuple[float, float, float, float]) -> List[List[Coordinate]]:
        pieces: List[List[Coordinate]] = []
        current: List[Coordinate] = []
        for start, end in zip(points, points[1:]):
            segment = cls._clip_segment(start, end, bbox)
            if segment is None:
                if len(current) >= 2:
                    pieces.append(current)
                current = []
                continue
            if not current:
                current = segment
            elif math.dist(current[-1], segment[0]) < 1e-10:
                current.append(segment[1])
            else:
                if len(current) >= 2:
                    pieces.append(current)
                current = segment
        if len(current) >= 2:
            pieces.append(current)
        return pieces

    @staticmethod
    def _clip_ring(points: Sequence[Coordinate], bbox: Tuple[float, float, float, float]) -> Optional[List[Coordinate]]:
        south, west, north, east = bbox
        ring = list(points)
        if ring and ring[0] == ring[-1]:
            ring = ring[:-1]

        def clip_edge(values: List[Coordinate], inside, intersection) -> List[Coordinate]:
            if not values:
                return []
            result: List[Coordinate] = []
            previous = values[-1]
            previous_inside = inside(previous)
            for current in values:
                current_inside = inside(current)
                if current_inside != previous_inside:
                    result.append(intersection(previous, current))
                if current_inside:
                    result.append(current)
                previous = current
                previous_inside = current_inside
            return result

        def vertical(value: float):
            def intersect(a: Coordinate, b: Coordinate) -> Coordinate:
                if abs(b[0] - a[0]) < 1e-12:
                    return value, a[1]
                t = (value - a[0]) / (b[0] - a[0])
                return value, a[1] + t * (b[1] - a[1])
            return intersect

        def horizontal(value: float):
            def intersect(a: Coordinate, b: Coordinate) -> Coordinate:
                if abs(b[1] - a[1]) < 1e-12:
                    return a[0], value
                t = (value - a[1]) / (b[1] - a[1])
                return a[0] + t * (b[0] - a[0]), value
            return intersect

        ring = clip_edge(ring, lambda p: p[0] >= west, vertical(west))
        ring = clip_edge(ring, lambda p: p[0] <= east, vertical(east))
        ring = clip_edge(ring, lambda p: p[1] >= south, horizontal(south))
        ring = clip_edge(ring, lambda p: p[1] <= north, horizontal(north))
        if len(ring) < 3:
            return None
        ring.append(ring[0])
        return ring

    @classmethod
    def _clip_geometry(cls, geometry: Dict[str, Any], bbox: Tuple[float, float, float, float]) -> Optional[Dict[str, Any]]:
        geometry_type = geometry.get("type")
        coordinates = geometry.get("coordinates")
        if geometry_type == "Point":
            south, west, north, east = bbox
            lon, lat = coordinates
            return geometry if west <= lon <= east and south <= lat <= north else None
        if geometry_type == "LineString":
            pieces = cls._clip_line(coordinates, bbox)
            if not pieces:
                return None
            if len(pieces) == 1:
                return {"type": "LineString", "coordinates": pieces[0]}
            return {"type": "MultiLineString", "coordinates": pieces}
        if geometry_type == "Polygon":
            outer = cls._clip_ring(coordinates[0], bbox) if coordinates else None
            if not outer:
                return None
            return {"type": "Polygon", "coordinates": [outer]}
        return None

    @classmethod
    def _element_to_feature(cls, element: Dict[str, Any], bbox: Tuple[float, float, float, float]) -> Optional[Dict[str, Any]]:
        tags = element.get("tags") or {}
        category = cls._category(tags)
        element_type = element.get("type")
        if element_type == "node" and element.get("lat") is not None and element.get("lon") is not None:
            geometry = {"type": "Point", "coordinates": [float(element["lon"]), float(element["lat"])]}
        elif element_type == "way":
            geometry_points = element.get("geometry") or []
            coordinates = [[float(point["lon"]), float(point["lat"])] for point in geometry_points if "lon" in point and "lat" in point]
            if len(coordinates) < 2:
                return None
            is_polygon = len(coordinates) >= 4 and coordinates[0] == coordinates[-1]
            geometry = {
                "type": "Polygon" if is_polygon else "LineString",
                "coordinates": [coordinates] if is_polygon else coordinates,
            }
        else:
            return None

        clipped = cls._clip_geometry(geometry, bbox)
        if clipped is None:
            return None
        properties = dict(tags)
        properties.update({
            "category": category,
            "osm_type": element_type,
            "osm_id": element.get("id"),
            "name": tags.get("name") or tags.get("ref") or category,
        })
        return {"type": "Feature", "properties": properties, "geometry": clipped}

    def _query(self, bbox: Tuple[float, float, float, float], timeout_sec: float) -> Tuple[List[Dict[str, Any]], str, bool]:
        south, west, north, east = bbox
        bbox_text = f"{south:.6f},{west:.6f},{north:.6f},{east:.6f}"
        query = f"""
[out:json][timeout:20];
(
  way["building"]({bbox_text});
  way["highway"]({bbox_text});
  way["railway"]({bbox_text});
  way["waterway"]({bbox_text});
  way["natural"~"water|wetland|wood|forest|scrub|heath|grassland|meadow"]({bbox_text});
  way["landuse"~"forest|farmland|grass|meadow|orchard|vineyard|scrub|residential|industrial"]({bbox_text});
  way["power"]({bbox_text});
  way["barrier"]({bbox_text});
  node["amenity"~"fire_station|hospital|school|police"]({bbox_text});
  node["place"]({bbox_text});
  node["man_made"]({bbox_text});
);
out geom tags;
"""
        last_error: Optional[Exception] = None
        for endpoint in self.OVERPASS_ENDPOINTS:
            try:
                response = requests.post(
                    endpoint,
                    data={"data": query},
                    headers={"User-Agent": "FireMap/1.0 (operational map metadata)"},
                    timeout=timeout_sec,
                )
                response.raise_for_status()
                payload = response.json()
                elements = payload.get("elements", []) if isinstance(payload, dict) else []
                truncated = len(elements) > self.max_features
                return elements[: self.max_features], endpoint, truncated
            except (OSError, ValueError, requests.RequestException) as exc:
                last_error = exc
        raise RuntimeError(str(last_error or "Overpass request failed"))

    def fetch(
        self,
        center_lat: float,
        center_lon: float,
        domain_size_m: float,
        timeout_sec: float = 12.0,
    ) -> Dict[str, Any]:
        """Return a clipped GeoJSON FeatureCollection for the active map."""
        key = self._cache_key(center_lat, center_lon, domain_size_m)
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(key)
            if cached and cached[0] > now:
                return cached[1]

            bbox = self._bbox(center_lat, center_lon, domain_size_m)
            try:
                elements, endpoint, truncated = self._query(bbox, timeout_sec)
                features = []
                for element in elements:
                    feature = self._element_to_feature(element, bbox)
                    if feature is not None:
                        features.append(feature)
                categories: Dict[str, int] = {}
                for feature in features:
                    category = feature["properties"].get("category", "other")
                    categories[category] = categories.get(category, 0) + 1
                result = {
                    "type": "FeatureCollection",
                    "source": "OpenStreetMap Overpass",
                    "source_endpoint": endpoint,
                    "is_live_api": True,
                    "center": {"lat": float(center_lat), "lon": float(center_lon)},
                    "bbox": [bbox[1], bbox[0], bbox[3], bbox[2]],
                    "domain_size_m": float(domain_size_m),
                    "feature_count": len(features),
                    "truncated": truncated,
                    "categories": categories,
                    "features": features,
                }
                self._cache[key] = (now + self.cache_ttl_sec, result)
                return result
            except Exception as exc:
                result = {
                    "type": "FeatureCollection",
                    "source": "OpenStreetMap Overpass unavailable",
                    "is_live_api": False,
                    "center": {"lat": float(center_lat), "lon": float(center_lon)},
                    "bbox": [bbox[1], bbox[0], bbox[3], bbox[2]],
                    "domain_size_m": float(domain_size_m),
                    "feature_count": 0,
                    "truncated": False,
                    "categories": {},
                    "features": [],
                    "error": str(exc)[:240],
                }
                self._cache[key] = (now + min(30.0, self.cache_ttl_sec), result)
                return result
