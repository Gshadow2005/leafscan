"""
LeafScan API — local test script
Run:  python app/test_api.py
Requires: pip install requests Pillow
"""

import sys
import json
import io
import requests
from PIL import Image, ImageDraw

BASE_URL = "http://localhost:8000"

PASS = "\033[92m✅ PASS\033[0m"
FAIL = "\033[91m❌ FAIL\033[0m"
INFO = "\033[94mℹ️ \033[0m"

def make_dummy_image(color=(34, 139, 34), size=(224, 224), fmt="JPEG") -> bytes:
    """Creates a small solid-color image in memory."""
    img = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 174, 174], fill=(0, 100, 0))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_root():
    print("\n── GET / ─────────────────────────────────────")
    r = requests.get(f"{BASE_URL}/")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert "service" in data
    assert data["service"] == "LeafScan API"
    print(f"{PASS}  status=200  model_loaded={data['model_loaded']}  classes={data['classes']}")


def test_health():
    print("\n── GET /health ───────────────────────────────")
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    print(f"{PASS}  status=ok  model_ready={data['model_ready']}")


def test_classes():
    print("\n── GET /classes ──────────────────────────────")
    r = requests.get(f"{BASE_URL}/classes")
    assert r.status_code == 200
    data = r.json()
    assert "classes" in data
    assert data["count"] > 0
    print(f"{PASS}  {data['count']} classes returned")
    for i, name in enumerate(data["classes"], 1):
        print(f"   {i:>2}. {name}")


def test_predict_jpeg():
    print("\n── POST /predict  (JPEG) ─────────────────────")
    img_bytes = make_dummy_image(fmt="JPEG")
    r = requests.post(
        f"{BASE_URL}/predict",
        files={"file": ("leaf.jpg", img_bytes, "image/jpeg")},
    )
    assert r.status_code == 200, f"Expected 200, got {r.status_code}\n{r.text}"
    data = r.json()
    assert "disease" in data
    assert "confidence" in data
    assert "all_scores" in data
    print(f"{PASS}  disease='{data['disease']}'  confidence={data['confidence']:.1f}%")
    print(f"   Top 3 scores:")
    top3 = sorted(data["all_scores"].items(), key=lambda x: x[1], reverse=True)[:3]
    for name, score in top3:
        print(f"     {name:<30} {score:.2f}%")


def test_predict_png():
    print("\n── POST /predict  (PNG) ──────────────────────")
    img_bytes = make_dummy_image(fmt="PNG")
    r = requests.post(
        f"{BASE_URL}/predict",
        files={"file": ("leaf.png", img_bytes, "image/png")},
    )
    assert r.status_code == 200, f"Expected 200, got {r.status_code}\n{r.text}"
    data = r.json()
    print(f"{PASS}  disease='{data['disease']}'  confidence={data['confidence']:.1f}%")


def test_predict_bad_type():
    print("\n── POST /predict  (bad file type — should 400) ─")
    r = requests.post(
        f"{BASE_URL}/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert r.status_code == 400, f"Expected 400, got {r.status_code}"
    print(f"{PASS}  Got 400 as expected: {r.json()['detail'][:60]}")


def test_predict_no_file():
    print("\n── POST /predict  (no file — should 422) ────")
    r = requests.post(f"{BASE_URL}/predict")
    assert r.status_code == 422, f"Expected 422, got {r.status_code}"
    print(f"{PASS}  Got 422 as expected (missing required field)")


def test_docs():
    print("\n── GET /docs ─────────────────────────────────")
    r = requests.get(f"{BASE_URL}/docs")
    assert r.status_code == 200
    print(f"{PASS}  Swagger UI available at {BASE_URL}/docs")


if __name__ == "__main__":
    print("=" * 50)
    print("  LeafScan API — Test Suite")
    print(f"  Target: {BASE_URL}")
    print("=" * 50)

    tests = [
        test_root,
        test_health,
        test_classes,
        test_predict_jpeg,
        test_predict_png,
        test_predict_bad_type,
        test_predict_no_file,
        test_docs,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"{FAIL}  {test.__name__}: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 50)

    if failed:
        sys.exit(1)