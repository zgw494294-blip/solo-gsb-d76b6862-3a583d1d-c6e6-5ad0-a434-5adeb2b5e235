"""片段另存（clip）接口测试：裁剪平移、原轨道保持、原子提交、错误码与并发。"""
import threading

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


def source_payload():
    """s1[0,1000)A s2[800,3000)B锁2 s3[1000,2500)A s4[4000,5000)C。

    分轨结果：s1→1、s2→2（锁定）、s3→1、s4→1，changes=0。
    """
    return {
        "track_count": 2,
        "expected_revision": 0,
        "subtitles": [
            {"id": "s1", "speaker": "Alice", "start_ms": 0, "end_ms": 1000},
            {"id": "s2", "speaker": "Bob", "start_ms": 800, "end_ms": 3000,
             "locked_track": 2},
            {"id": "s3", "speaker": "Alice", "start_ms": 1000, "end_ms": 2500},
            {"id": "s4", "speaker": "Carol", "start_ms": 4000, "end_ms": 5000},
        ],
    }


def make_source(client, scene_id="scene-src"):
    resp = client.put(f"/scenes/{scene_id}", json=source_payload())
    assert resp.status_code == 200
    assert resp.json()["revision"] == 1
    return resp.json()


def clip_body(target="clip-1", from_ms=500, to_ms=2000, expected_revision=1):
    return {
        "expected_revision": expected_revision,
        "target_scene_id": target,
        "from_ms": from_ms,
        "to_ms": to_ms,
    }


# ---------------------------------------------------------------------------
# 裁剪与平移
# ---------------------------------------------------------------------------


def test_clip_crops_shifts_and_keeps_original_tracks():
    client = TestClient(app)
    make_source(client)

    resp = client.post("/scenes/scene-src/clips", json=clip_body())
    assert resp.status_code == 200
    body = resp.json()
    assert body["feasible"] is True
    assert body["scene_id"] == "clip-1"
    assert body["revision"] == 1
    assert body["track_count"] == 2  # 沿用源 track_count
    # s4[4000,5000) 不相交被丢弃；其余裁到 [500,2000) 内再减去 500
    assert body["subtitles"] == [
        {"id": "s1", "speaker": "Alice", "start_ms": 0, "end_ms": 500,
         "locked_track": 1},
        {"id": "s2", "speaker": "Bob", "start_ms": 300, "end_ms": 1500,
         "locked_track": 2},
        {"id": "s3", "speaker": "Alice", "start_ms": 500, "end_ms": 1500,
         "locked_track": 1},
    ]
    # 源分轨写入 locked_track，重算后保持原画面轨道
    assert body["assignments"] == [
        {"id": "s1", "track": 1},
        {"id": "s2", "track": 2},
        {"id": "s3", "track": 1},
    ]
    assert body["changes"] == 0

    # 目标可经既有读取接口取得同一快照
    got = client.get("/scenes/clip-1")
    assert got.status_code == 200
    assert got.json() == body


def test_clip_window_is_half_open():
    """端点相接不算相交：end_ms == from_ms 与 start_ms == to_ms 都被排除。"""
    client = TestClient(app)
    make_source(client)

    # s1[0,1000) 在 1000 结束、s4[4000,5000) 在 4000 开始，均不相交
    resp = client.post(
        "/scenes/scene-src/clips",
        json=clip_body(target="clip-half", from_ms=1000, to_ms=4000),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert [s["id"] for s in body["subtitles"]] == ["s2", "s3"]
    assert body["subtitles"][0]["start_ms"] == 0     # 800 裁到 1000 → 0
    assert body["subtitles"][0]["end_ms"] == 2000    # 3000 → 2000
    assert body["subtitles"][1]["start_ms"] == 0     # 1000 → 0
    assert body["subtitles"][1]["end_ms"] == 1500    # 2500 → 1500


def test_clip_recomputes_changes_for_remaining_chain():
    """changes 按片段内剩余的说话人相邻链重算，而非沿用源场景的数值。"""
    client = TestClient(app)
    # a1→1（锁定）、a2→2（锁定）迫使一次换轨；a3 与 a2 同轨，源 changes=1
    resp = client.put("/scenes/scene-chain", json={
        "track_count": 2,
        "expected_revision": 0,
        "subtitles": [
            {"id": "a1", "speaker": "A", "start_ms": 0, "end_ms": 100,
             "locked_track": 1},
            {"id": "a2", "speaker": "A", "start_ms": 100, "end_ms": 200,
             "locked_track": 2},
            {"id": "a3", "speaker": "A", "start_ms": 200, "end_ms": 300},
        ],
    })
    assert resp.json()["changes"] == 1

    # 窗口 [150,300) 只剩 a2、a3（同轨 2）：片段内换轨 0 次
    resp = client.post(
        "/scenes/scene-chain/clips",
        json=clip_body(target="clip-chain", from_ms=150, to_ms=300),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["subtitles"] == [
        {"id": "a2", "speaker": "A", "start_ms": 0, "end_ms": 50,
         "locked_track": 2},
        {"id": "a3", "speaker": "A", "start_ms": 50, "end_ms": 150,
         "locked_track": 2},
    ]
    assert body["assignments"] == [
        {"id": "a2", "track": 2},
        {"id": "a3", "track": 2},
    ]
    assert body["changes"] == 0


def test_clip_of_clip_uses_target_as_new_source():
    """片段本身是普通场景：可作为源再次裁剪（expected_revision=1）。"""
    client = TestClient(app)
    make_source(client)
    client.post("/scenes/scene-src/clips", json=clip_body())  # clip-1 rev 1

    resp = client.post(
        "/scenes/clip-1/clips",
        json=clip_body(target="clip-2", from_ms=0, to_ms=400,
                       expected_revision=1),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["revision"] == 1
    # clip-1 中 s1[0,500)→[0,400)，s2[300,1500)→[300,400)；s3 从 500 开始被排除
    assert [(s["id"], s["start_ms"], s["end_ms"]) for s in body["subtitles"]] == [
        ("s1", 0, 400),
        ("s2", 300, 400),
    ]


# ---------------------------------------------------------------------------
# 422：窗口与字段校验
# ---------------------------------------------------------------------------


def test_clip_no_intersecting_subtitles_422_and_no_target_left():
    client = TestClient(app)
    make_source(client)

    # s2 在 3000 结束、s4 从 4000 开始：[3000,4000) 内没有任何字幕
    resp = client.post(
        "/scenes/scene-src/clips",
        json=clip_body(target="clip-empty", from_ms=3000, to_ms=4000),
    )
    assert resp.status_code == 422
    # 失败不留下目标，源场景不变
    assert client.get("/scenes/clip-empty").status_code == 404
    assert client.get("/scenes/scene-src").json()["revision"] == 1


def test_clip_invalid_window_422():
    client = TestClient(app)
    make_source(client)

    for from_ms, to_ms in [(2000, 2000), (3000, 2000), (-1, 100), (0, -5)]:
        resp = client.post(
            "/scenes/scene-src/clips",
            json=clip_body(from_ms=from_ms, to_ms=to_ms),
        )
        assert resp.status_code == 422, (from_ms, to_ms)
    assert client.get("/scenes/clip-1").status_code == 404


def test_clip_time_fields_reject_bool_and_string():
    client = TestClient(app)
    make_source(client)

    bad_values = [True, False, "500", "2000", 1.5, 1.0, None]
    for field in ("from_ms", "to_ms"):
        for value in bad_values:
            body = clip_body()
            body[field] = value
            resp = client.post("/scenes/scene-src/clips", json=body)
            assert resp.status_code == 422, (field, value)
    assert client.get("/scenes/clip-1").status_code == 404


def test_clip_blank_target_scene_id_422():
    client = TestClient(app)
    make_source(client)

    for target in ("", "   "):
        body = clip_body(target=target)
        resp = client.post("/scenes/scene-src/clips", json=body)
        assert resp.status_code == 422


def test_clip_missing_or_invalid_expected_revision_422():
    client = TestClient(app)
    make_source(client)

    body = clip_body()
    del body["expected_revision"]
    assert client.post("/scenes/scene-src/clips", json=body).status_code == 422

    body = clip_body(expected_revision=-1)
    assert client.post("/scenes/scene-src/clips", json=body).status_code == 422


def test_clip_blank_source_scene_id_422():
    client = TestClient(app)
    resp = client.post("/scenes/%20%20/clips", json=clip_body())
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 404 / 409
# ---------------------------------------------------------------------------


def test_clip_source_not_found_404():
    client = TestClient(app)
    resp = client.post("/scenes/no-such/clips", json=clip_body())
    assert resp.status_code == 404
    assert client.get("/scenes/clip-1").status_code == 404


def test_clip_stale_revision_409_and_nothing_changes():
    client = TestClient(app)
    make_source(client)  # 当前 revision=1

    for stale in (0, 2):
        resp = client.post(
            "/scenes/scene-src/clips",
            json=clip_body(expected_revision=stale),
        )
        assert resp.status_code == 409
        assert resp.json()["detail"]["current_revision"] == 1

    # 失败不留下目标，源场景仍是 revision 1 的原内容
    assert client.get("/scenes/clip-1").status_code == 404
    got = client.get("/scenes/scene-src").json()
    assert got["revision"] == 1
    assert [s["id"] for s in got["subtitles"]] == ["s1", "s2", "s3", "s4"]


def test_clip_existing_target_409_and_target_untouched():
    client = TestClient(app)
    make_source(client)
    client.post("/scenes/scene-src/clips", json=clip_body())  # clip-1 已存在

    # 同一目标再裁剪：409；换一个源也一样
    make_source(client, "scene-src2")
    resp = client.post("/scenes/scene-src2/clips", json=clip_body())
    assert resp.status_code == 409
    assert resp.json()["detail"]["scene_id"] == "clip-1"

    # 已有目标内容未被覆盖，源场景版本不变
    got = client.get("/scenes/clip-1").json()
    assert got["revision"] == 1
    assert [s["id"] for s in got["subtitles"]] == ["s1", "s2", "s3"]
    assert client.get("/scenes/scene-src2").json()["revision"] == 1


def test_clip_target_equals_source_is_409():
    client = TestClient(app)
    make_source(client)
    resp = client.post(
        "/scenes/scene-src/clips", json=clip_body(target="scene-src")
    )
    assert resp.status_code == 409
    assert client.get("/scenes/scene-src").json()["revision"] == 1


# ---------------------------------------------------------------------------
# 目标场景可用既有接口；源场景与播放会话不受影响
# ---------------------------------------------------------------------------


def test_clip_target_supports_read_revise_and_playback():
    client = TestClient(app)
    make_source(client)
    client.post("/scenes/scene-src/clips", json=clip_body())

    # 修订片段：expected_revision=1 → revision 2
    revise = {
        "track_count": 2,
        "expected_revision": 1,
        "subtitles": [
            {"id": "n1", "speaker": "N", "start_ms": 0, "end_ms": 100},
        ],
    }
    resp = client.put("/scenes/clip-1", json=revise)
    assert resp.status_code == 200
    assert resp.json()["revision"] == 2

    # 播放片段：会话从 0 毫秒起，活动字幕按片段时间
    resp = client.post("/sessions", json={"scene_id": "clip-1"})
    assert resp.status_code == 200
    session = resp.json()
    assert session["source_revision"] == 2
    assert session["cursor_ms"] == 0
    resp = client.post(
        f"/sessions/{session['session_id']}/advance", json={"time_ms": 50}
    )
    assert resp.status_code == 200
    assert [a["id"] for a in resp.json()["active"]] == ["n1"]


def test_clip_does_not_touch_source_or_playback_sessions():
    client = TestClient(app)
    make_source(client)
    session = client.post("/sessions", json={"scene_id": "scene-src"}).json()
    client.post(
        f"/sessions/{session['session_id']}/advance", json={"time_ms": 900}
    )

    resp = client.post("/scenes/scene-src/clips", json=clip_body())
    assert resp.status_code == 200

    # 源场景不变
    got = client.get("/scenes/scene-src").json()
    assert got["revision"] == 1
    assert [s["id"] for s in got["subtitles"]] == ["s1", "s2", "s3", "s4"]

    # 源场景上的播放会话不受影响：游标、源版本与活动字幕保持原样
    resp = client.post(
        f"/sessions/{session['session_id']}/advance", json={"time_ms": 1000}
    )
    assert resp.status_code == 200
    state = resp.json()
    assert state["source_revision"] == 1
    assert state["cursor_ms"] == 1000
    assert [a["id"] for a in state["active"]] == ["s2", "s3"]


def test_clip_persists_across_store_reopen():
    """模拟容器重启：同一数据库路径重新打开后片段仍可读。"""
    import os
    from app.storage import SceneStore

    db_path = os.environ["SCENES_DB_PATH"]
    client = TestClient(app)
    make_source(client)
    client.post("/scenes/scene-src/clips", json=clip_body())

    reopened = SceneStore(db_path)
    record = reopened.get_scene("clip-1")
    assert record is not None
    assert record.revision == 1
    assert record.track_count == 2
    assert [(s["id"], s["start_ms"], s["end_ms"], s["locked_track"])
            for s in record.subtitles] == [
        ("s1", 0, 500, 1),
        ("s2", 300, 1500, 2),
        ("s3", 500, 1500, 1),
    ]
    assert record.assignments == [
        {"id": "s1", "track": 1},
        {"id": "s2", "track": 2},
        {"id": "s3", "track": 1},
    ]


# ---------------------------------------------------------------------------
# 并发：核对源版本与创建目标是一次提交
# ---------------------------------------------------------------------------


def test_concurrent_clips_same_target_only_one_succeeds():
    client = TestClient(app)
    make_source(client)

    results = []
    barrier = threading.Barrier(4)

    def worker(i):
        local = TestClient(app)
        barrier.wait()
        resp = local.post(
            "/scenes/scene-src/clips",
            json=clip_body(target="clip-race", from_ms=500 + i, to_ms=2000),
        )
        results.append(resp.status_code)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count(200) == 1
    assert results.count(409) == 3
    # 目标只被唯一获胜者写入一次，版本为 1
    assert client.get("/scenes/clip-race").json()["revision"] == 1
    # 源场景不被片段提交修改
    assert client.get("/scenes/scene-src").json()["revision"] == 1


def test_concurrent_clips_distinct_targets_all_succeed():
    """片段对源场景只读：同一源版本并发出多个片段互不冲突。"""
    client = TestClient(app)
    make_source(client)

    results = []
    barrier = threading.Barrier(3)

    def worker(i):
        local = TestClient(app)
        barrier.wait()
        resp = local.post(
            "/scenes/scene-src/clips", json=clip_body(target=f"clip-{i}")
        )
        results.append(resp.status_code)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results == [200, 200, 200]
    for i in range(3):
        assert client.get(f"/scenes/clip-{i}").json()["revision"] == 1


def test_concurrent_clip_and_source_revision_are_serialized():
    """修订与片段共用同一把写锁：片段要么基于修订前版本成功，要么 409。

    无论哪种交错，源场景都推进到 revision 2，且若片段成功，其内容
    一定来自 revision 1 的快照（核对与创建在一次提交内完成）。
    """
    client = TestClient(app)
    make_source(client)

    barrier = threading.Barrier(2)
    outcomes = {}

    def do_clip():
        local = TestClient(app)
        barrier.wait()
        resp = local.post("/scenes/scene-src/clips", json=clip_body())
        outcomes["clip"] = resp.status_code

    def do_revise():
        local = TestClient(app)
        body = source_payload()
        body["expected_revision"] = 1
        body["subtitles"][3]["end_ms"] = 6000
        barrier.wait()
        resp = local.put("/scenes/scene-src", json=body)
        outcomes["revise"] = resp.status_code

    threads = [threading.Thread(target=do_clip), threading.Thread(target=do_revise)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 片段不改源场景，修订一定成功
    assert outcomes["revise"] == 200
    assert client.get("/scenes/scene-src").json()["revision"] == 2
    # 片段：成功则内容来自 revision 1 快照；失败则必是 409 且不留目标
    if outcomes["clip"] == 200:
        got = client.get("/scenes/clip-1").json()
        assert got["revision"] == 1
        assert [s["id"] for s in got["subtitles"]] == ["s1", "s2", "s3"]
    else:
        assert outcomes["clip"] == 409
        assert client.get("/scenes/clip-1").status_code == 404
