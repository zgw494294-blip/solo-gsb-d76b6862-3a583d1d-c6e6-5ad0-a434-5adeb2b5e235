"""场景持久化：本地 SQLite。

一个场景（scene）保存某次成功修订的完整快照：
字幕输入、分轨结果、换轨次数与版本号（revision）。

并发语义采用乐观锁：客户端提交 expected_revision，新场景填 0，
已存在场景须填写当前版本；服务端在一个 BEGIN IMMEDIATE 事务中
读取当前版本并写入新版本，因此同一版本的并发提交最多一次成功，
其余全部以 RevisionConflict 失败且不改动已有数据。

片段另存（commit_clip）同样在一个 BEGIN IMMEDIATE 事务中完成：
核对源场景当前版本与 expected_revision 一致、确认目标 scene_id
未被占用，然后以 revision=1 写入目标场景。源场景只读不写；
任一步失败都整体回滚，不留下目标场景。
"""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_DB_PATH = os.environ.get("SCENES_DB_PATH", "scenes.db")


class RevisionConflict(Exception):
    """expected_revision 与库中当前版本不一致。"""

    def __init__(self, scene_id: str, current_revision: int) -> None:
        self.scene_id = scene_id
        self.current_revision = current_revision
        super().__init__(
            f"场景 {scene_id!r} 的版本冲突：期望新版本基于 "
            f"revision={current_revision} 再次提交"
        )


class SceneNotFound(Exception):
    """片段另存提交时源场景不存在（对应 HTTP 404）。"""

    def __init__(self, scene_id: str) -> None:
        self.scene_id = scene_id
        super().__init__(f"场景 {scene_id!r} 不存在")


class TargetSceneExists(Exception):
    """片段另存提交时目标 scene_id 已被占用（对应 HTTP 409）。"""

    def __init__(self, scene_id: str) -> None:
        self.scene_id = scene_id
        super().__init__(f"目标场景 {scene_id!r} 已存在")


@dataclass
class SceneRecord:
    scene_id: str
    revision: int
    track_count: int
    subtitles: List[Dict[str, Any]]
    assignments: List[Dict[str, Any]]
    changes: int


class SceneStore:
    """SQLite 场景存储。线程安全：每次操作使用独立连接（check_same_thread=False）。"""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        parent = os.path.dirname(os.path.abspath(db_path))
        os.makedirs(parent, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # WAL 允许读写并发；busy_timeout 在写锁竞争时等待而非立即报错
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scenes (
                    scene_id      TEXT PRIMARY KEY,
                    revision      INTEGER NOT NULL,
                    track_count   INTEGER NOT NULL,
                    subtitles     TEXT    NOT NULL,
                    assignments   TEXT    NOT NULL,
                    changes       INTEGER NOT NULL,
                    updated_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                )
                """
            )

    def commit_revision(
        self,
        scene_id: str,
        expected_revision: int,
        track_count: int,
        subtitles: List[Dict[str, Any]],
        assignments: List[Dict[str, Any]],
        changes: int,
    ) -> int:
        """以乐观锁写入一次成功修订，返回新版本号。

        新场景（expected_revision == 0 且库中不存在）写入 revision=1；
        已存在场景仅当 expected_revision == 当前 revision 时覆盖为 +1；
        否则抛出 RevisionConflict，且不改动任何已有数据。
        """
        conn = self._connect()
        try:
            # IMMEDIATE 立即获取保留写锁，串行化并发提交：
            # 同一 expected_revision 的并发提交只有第一个能通过版本核对。
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT revision FROM scenes WHERE scene_id = ?", (scene_id,)
            ).fetchone()

            if row is None:
                if expected_revision != 0:
                    raise RevisionConflict(scene_id, 0)
                new_revision = 1
                conn.execute(
                    """
                    INSERT INTO scenes
                        (scene_id, revision, track_count, subtitles, assignments, changes)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        scene_id,
                        new_revision,
                        track_count,
                        json.dumps(subtitles, ensure_ascii=False),
                        json.dumps(assignments, ensure_ascii=False),
                        changes,
                    ),
                )
            else:
                current_revision = row["revision"]
                if expected_revision != current_revision:
                    raise RevisionConflict(scene_id, current_revision)
                new_revision = current_revision + 1
                conn.execute(
                    """
                    UPDATE scenes
                       SET revision = ?, track_count = ?, subtitles = ?,
                           assignments = ?, changes = ?,
                           updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
                     WHERE scene_id = ?
                    """,
                    (
                        new_revision,
                        track_count,
                        json.dumps(subtitles, ensure_ascii=False),
                        json.dumps(assignments, ensure_ascii=False),
                        changes,
                        scene_id,
                    ),
                )
            conn.commit()
            return new_revision
        except RevisionConflict:
            conn.rollback()
            raise
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def commit_clip(
        self,
        source_scene_id: str,
        expected_revision: int,
        target_scene_id: str,
        track_count: int,
        subtitles: List[Dict[str, Any]],
        assignments: List[Dict[str, Any]],
        changes: int,
    ) -> int:
        """核对源场景版本并创建片段场景，作为一次提交；返回目标版本（恒为 1）。

        在同一个 BEGIN IMMEDIATE 事务中：读取源场景并核对
        expected_revision 与其当前版本一致、确认 target_scene_id 未被
        占用，然后以 revision=1 写入目标场景。源场景只读不写；任一步
        失败都整体回滚——不留下目标场景，也不改动源场景。因此并发下
        同一目标 scene_id 的片段提交最多一次成功，源场景在提交期间被
        修订也会让本次提交以 RevisionConflict 失败。
        """
        conn = self._connect()
        try:
            # IMMEDIATE 立即获取保留写锁：版本核对、目标占用检查与写入
            # 三者串行化，不会出现“核对通过后被别的提交抢先”的交错。
            conn.execute("BEGIN IMMEDIATE")
            src = conn.execute(
                "SELECT revision FROM scenes WHERE scene_id = ?",
                (source_scene_id,),
            ).fetchone()
            if src is None:
                raise SceneNotFound(source_scene_id)
            current_revision = src["revision"]
            if expected_revision != current_revision:
                raise RevisionConflict(source_scene_id, current_revision)
            tgt = conn.execute(
                "SELECT 1 FROM scenes WHERE scene_id = ?", (target_scene_id,)
            ).fetchone()
            if tgt is not None:
                raise TargetSceneExists(target_scene_id)
            conn.execute(
                """
                INSERT INTO scenes
                    (scene_id, revision, track_count, subtitles, assignments, changes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    target_scene_id,
                    1,
                    track_count,
                    json.dumps(subtitles, ensure_ascii=False),
                    json.dumps(assignments, ensure_ascii=False),
                    changes,
                ),
            )
            conn.commit()
            return 1
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_scene(self, scene_id: str) -> Optional[SceneRecord]:
        """按 scene_id 读取最近一次成功修订；不存在返回 None。

        单行单行快照读取（WAL 下一条 SELECT 对同一行取一致版本），
        不会读到某次提交的部分字段。
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT scene_id, revision, track_count, subtitles, assignments, changes
                  FROM scenes WHERE scene_id = ?
                """,
                (scene_id,),
            ).fetchone()
        if row is None:
            return None
        subtitles, assignments = _loads_pair(row["subtitles"], row["assignments"])
        return SceneRecord(
            scene_id=row["scene_id"],
            revision=row["revision"],
            track_count=row["track_count"],
            subtitles=subtitles,
            assignments=assignments,
            changes=row["changes"],
        )


def _loads_pair(subtitles_json: str, assignments_json: str) -> Tuple[List, List]:
    """同源反序列化，保证读取到的字幕与分轨属于同一次提交。"""
    subtitles: List[Dict[str, Any]] = json.loads(subtitles_json)
    assignments: List[Dict[str, Any]] = json.loads(assignments_json)
    return subtitles, assignments
