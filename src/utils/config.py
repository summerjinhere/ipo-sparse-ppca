"""
utils/config.py
───────────────
Config 로더 + 로거 설정

설계 원칙:
  - YAML 설정을 dot-notation으로 접근 가능
  - 환경변수로 민감정보 분리 (.env)
  - 모든 모듈이 동일한 로거 사용
"""

import yaml
import os
from pathlib import Path
from typing import Any
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# ── 프로젝트 루트 경로 ─────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent


# ── Dot-notation Config 클래스 ─────────────────────────────
class Config:
    """
    YAML 설정을 dot-notation으로 접근

    사용법:
        cfg = Config.load()
        print(cfg.model.ppca.k_max)   # 8
        print(cfg.data.sample.start_year)  # 2018
    """

    def __init__(self, d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            elif isinstance(v, list):
                setattr(self, k, [
                    Config(i) if isinstance(i, dict) else i
                    for i in v
                ])
            else:
                setattr(self, k, v)

    def __repr__(self):
        return f"Config({self.__dict__})"

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        """
        YAML 파일 로드.
        path 미지정 시 configs/config.yaml 기본 사용.
        """
        if path is None:
            path = ROOT / "configs" / "config.yaml"
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(raw)

    def to_dict(self) -> dict:
        """Config → dict 변환 (직렬화용)"""
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Config):
                result[k] = v.to_dict()
            elif isinstance(v, list):
                result[k] = [
                    i.to_dict() if isinstance(i, Config) else i
                    for i in v
                ]
            else:
                result[k] = v
        return result


# ── 로거 설정 ──────────────────────────────────────────────
def setup_logger(
    log_file: str | Path | None = None,
    level: str = "INFO",
) -> None:
    """
    loguru 로거 설정.

    Args:
        log_file: 파일 경로 (None이면 콘솔만)
        level:    로그 레벨 (DEBUG / INFO / WARNING / ERROR)
    """
    logger.remove()  # 기본 핸들러 제거

    # 콘솔 출력
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level=level,
        colorize=True,
    )

    # 파일 출력
    if log_file is not None:
        log_path = ROOT / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            sink=str(log_path),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} | {message}",
            level=level,
            rotation="10 MB",
            retention="30 days",
            encoding="utf-8",
        )


# ── 환경변수 헬퍼 ─────────────────────────────────────────
def get_env(key: str, default: Any = None) -> Any:
    """
    환경변수 조회. 없으면 default 반환.

    .env 파일 예시:
        OPENAI_API_KEY=sk-...
        DB_URL=postgresql://user:pass@localhost/ipo_db
        CHROMA_HOST=localhost
    """
    val = os.getenv(key, default)
    if val is None:
        logger.warning(f"환경변수 '{key}'가 설정되지 않음")
    return val
