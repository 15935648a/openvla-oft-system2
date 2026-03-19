import requests

BASE = "http://localhost:8000"


def test_health():
    r = requests.get(f"{BASE}/health")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    assert r.json() == {"status": "ok"}, f"Unexpected body: {r.json()}"
    print("[PASS] GET /health ->", r.json())


def test_subgoal():
    payload = {
        "task": "pick up the red block and place it on the green plate",
        "summary": "The robot gripper is OPEN (holding nothing). I see a red block and a green plate.",
    }
    r = requests.post(f"{BASE}/subgoal", json=payload)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    body = r.json()
    assert "subgoal" in body, f"Missing 'subgoal' key: {body}"
    assert isinstance(body["subgoal"], str) and body["subgoal"], f"Empty subgoal: {body}"
    print("[PASS] POST /subgoal ->", body)


if __name__ == "__main__":
    test_health()
    test_subgoal()
    print("\nAll tests passed.")
