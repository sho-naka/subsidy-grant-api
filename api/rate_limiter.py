import time
import threading
import os
from collections import deque
from typing import Tuple


class SimpleSlidingWindow:
    """グローバル（全体）カウントのスライディングウィンドウ実装。

    全リクエストを1つの deque に格納して、過去 60 秒間の全リクエスト数をカウントします。
    非常にシンプルで単一プロセス向けです。
    """

    def __init__(self, per_min: int):
        self.per_min = int(per_min)
        self.window_seconds = 60
        self.deque = deque()
        self.lock = threading.Lock()

    def allow(self) -> Tuple[bool, int]:
        """全体のリクエストを許可するか判定する。戻り値は (allowed, remaining)。"""
        now = time.time()
        cutoff = now - self.window_seconds
        with self.lock:
            # 古いタイムスタンプを削除
            while self.deque and self.deque[0] <= cutoff:
                self.deque.popleft()

            if len(self.deque) < self.per_min:
                self.deque.append(now)
                remaining = self.per_min - len(self.deque)
                return True, remaining
            else:
                return False, 0


# グローバルインスタンスを環境変数で設定
_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
_limiter = SimpleSlidingWindow(_PER_MIN)


def allow_request(_: str = "global") -> Tuple[bool, int]:
    """引数は無視される（互換性のために ip を取るが、内部はグローバルカウントを使用）。"""
    return _limiter.allow()
