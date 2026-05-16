"""
ingestion/edgar_crawler.py
──────────────────────────
EDGAR에서 S-1 / 424B4 HTML을 수집하는 크롤러

파이프라인 위치:
  [EDGAR API] → CIK 조회 → Filing 필터링
              → HTML 다운로드 → 로컬 저장

설계 원칙:
  - SEC rate limit 준수 (9 req/sec)
  - 실패 시 exponential backoff retry
  - 이미 다운로드된 파일 skip (멱등성)
  - 진행상황 로그 + 중간 저장
"""

import os
import time
import json
import hashlib
import requests
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Iterator
from loguru import logger
from tqdm import tqdm

from src.utils.config import Config, ROOT


# ── 데이터 클래스 ────────────────────────────────────────
@dataclass
class FilingMeta:
    """
    개별 Filing의 메타데이터.

    하나의 IPO 기업에 S-1 여러 개 + 424B4 하나 존재.
    우리의 primary target = 424B4.
    """
    cik:          str         # SEC 기업 고유 ID
    company_name: str
    form_type:    str         # "S-1" / "424B4"
    filed_date:   str         # "2021-11-01"
    accession_no: str         # SEC accession number
    html_url:     str         # HTML 파일 직접 URL
    file_size_kb: float = 0.0
    downloaded:   bool  = False


# ── 메인 크롤러 ─────────────────────────────────────────
class EdgarCrawler:
    """
    EDGAR Full-Text Search API 기반 크롤러.

    사용법:
        cfg = Config.load()
        crawler = EdgarCrawler(cfg)

        # 1단계: Filing 목록 수집
        filings = list(crawler.iter_filings(
            form_types=["424B4", "S-1"],
            start="2018-01-01",
            end="2024-12-31",
        ))

        # 2단계: HTML 다운로드
        crawler.download_all(filings)
    """

    # SEC EDGAR 엔드포인트
    SEARCH_URL   = "https://efts.sec.gov/LATEST/search-index"
    ARCHIVES_URL = "https://www.sec.gov/Archives"

    # 제외 SIC 코드 (REIT, 특수목적법인 등)
    EXCLUDE_SIC = {
        6726,  # Investment Offices (SPAC 포함)
        6798,  # Real Estate Investment Trusts
        6199,  # Finance Services
    }

    def __init__(self, cfg: Config):
        self.cfg  = cfg
        self.rate = cfg.data.edgar.rate_limit_per_sec  # 9
        self.retry_max = cfg.data.edgar.retry_max       # 3

        # 저장 경로
        self.raw_dir = ROOT / cfg.paths.raw_html
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        # 세션 설정 (헤더 포함 — SEC 규정)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": os.getenv("SEC_USER_AGENT", "Research/1.0 your-email@example.com"),
            "Accept-Encoding": "gzip, deflate",
            "Host": "efts.sec.gov",
        })

        self._last_request_time = 0.0

    # ── Rate Limiting ───────────────────────────────────
    def _throttle(self):
        """
        SEC rate limit 준수: 초당 최대 9 req.
        두 요청 사이 최소 1/9초 대기.
        """
        elapsed = time.time() - self._last_request_time
        wait = (1.0 / self.rate) - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_time = time.time()

    # ── HTTP GET with Retry ─────────────────────────────
    def _get(self, url: str, **kwargs) -> requests.Response:
        """
        Exponential backoff retry가 적용된 GET 요청.

        왜 exponential backoff인가:
          - 일시적 서버 과부하에서 즉시 재시도는 역효과
          - 1초 → 2초 → 4초 간격으로 서버 회복 대기
        """
        self._throttle()
        for attempt in range(self.retry_max):
            try:
                resp = self.session.get(
                    url,
                    timeout=self.cfg.data.edgar.timeout_sec,
                    **kwargs
                )
                resp.raise_for_status()
                return resp

            except requests.HTTPError as e:
                if resp.status_code == 429:   # Too Many Requests
                    wait = 2 ** attempt * 5   # 5, 10, 20초
                    logger.warning(f"Rate limited. {wait}초 대기 후 재시도.")
                    time.sleep(wait)
                elif resp.status_code >= 500: # 서버 오류
                    wait = 2 ** attempt
                    logger.warning(f"서버 오류 {resp.status_code}. {wait}초 후 재시도.")
                    time.sleep(wait)
                else:
                    raise
            except requests.RequestException as e:
                wait = 2 ** attempt
                logger.warning(f"요청 실패: {e}. {wait}초 후 재시도.")
                time.sleep(wait)

        raise RuntimeError(f"최대 재시도({self.retry_max}회) 초과: {url}")

    # ── CIK 조회 ────────────────────────────────────────
    def get_cik(self, company_name: str) -> str | None:
        """
        회사명 → CIK 변환.

        EDGAR의 Company Search API 사용.
        반환: "0001720671" 형태의 10자리 CIK
        """
        url = "https://efts.sec.gov/LATEST/search-index"
        params = {
            "q": f'"{company_name}"',
            "dateRange": "custom",
            "forms": "424B4",
        }
        try:
            resp = self._get(url, params=params)
            hits = resp.json().get("hits", {}).get("hits", [])
            if hits:
                return hits[0]["_source"].get("entity_id", "").lstrip("0") or None
        except Exception as e:
            logger.debug(f"CIK 조회 실패 ({company_name}): {e}")
        return None

    # ── Filing 목록 수집 ─────────────────────────────────
    def iter_filings(
        self,
        form_types: list[str] = ["424B4", "S-1"],
        start:      str = "2018-01-01",
        end:        str = "2024-12-31",
        batch_size: int = 100,
    ) -> Iterator[FilingMeta]:
        """
        EDGAR Full-Text Search API로 Filing 목록 순회.

        구조:
          GET /LATEST/search-index
          ?forms=424B4&dateRange=custom
          &startdt=2018-01-01&enddt=2024-12-31
          &from=0&size=100

        페이지네이션: from 파라미터로 offset 이동

        Yields:
            FilingMeta 객체 (하나씩)
        """
        for form_type in form_types:
            offset = 0
            total = None

            while True:
                params = {
                    "forms":     form_type,
                    "dateRange": "custom",
                    "startdt":   start,
                    "enddt":     end,
                    "from":      offset,
                    "size":      batch_size,
                }

                try:
                    resp  = self._get(self.SEARCH_URL, params=params)
                    data  = resp.json()
                    hits  = data.get("hits", {})
                    items = hits.get("hits", [])

                    if total is None:
                        total = hits.get("total", {}).get("value", 0)
                        logger.info(f"[{form_type}] 총 {total}건 발견")

                    if not items:
                        break

                    for item in items:
                        src = item.get("_source", {})

                        # 기본 필터: 제외 조건
                        if self._should_exclude(src):
                            continue

                        filing = FilingMeta(
                            cik=src.get("entity_id", ""),
                            company_name=src.get("display_names", [""])[0],
                            form_type=form_type,
                            filed_date=src.get("file_date", ""),
                            accession_no=src.get("accession_no", ""),
                            html_url=self._build_html_url(src),
                        )
                        yield filing

                    offset += batch_size
                    if offset >= total:
                        break

                except Exception as e:
                    logger.error(f"Filing 목록 수집 오류 (offset={offset}): {e}")
                    break

    def _should_exclude(self, src: dict) -> bool:
        """
        제외 조건 확인.

        제외 대상:
          - SIC 코드가 EXCLUDE_SIC에 포함
          - "SPAC", "acquisition corp" 회사명 포함
          - foreign private issuer 표시

        반환:
          True → 제외, False → 포함
        """
        company = src.get("display_names", [""])[0].lower()
        sic     = int(src.get("period_of_report", "0") or "0")

        # REIT / 투자법인 SIC 제외
        if sic in self.EXCLUDE_SIC:
            return True

        # SPAC 키워드 필터
        spac_keywords = [
            "acquisition corp", "blank check",
            "special purpose", "spac"
        ]
        if any(kw in company for kw in spac_keywords):
            return True

        return False

    def _build_html_url(self, src: dict) -> str:
        """
        accession_no → 실제 HTML URL 변환.

        EDGAR URL 구조:
          /Archives/edgar/data/{cik}/{accession_no_no_dash}/{filename}

        accession_no: "0001720671-21-000001"
        → "0001720671/000172067121000001"
        """
        cik   = src.get("entity_id", "")
        acc   = src.get("accession_no", "").replace("-", "")
        files = src.get("file_info", [{}])
        fname = files[0].get("file_name", "") if files else ""

        if cik and acc and fname:
            cik_folder = acc[:10]
            return f"{self.ARCHIVES_URL}/edgar/data/{cik}/{cik_folder}/{acc}/{fname}"
        return ""

    # ── HTML 다운로드 ────────────────────────────────────
    def download_all(
        self,
        filings: list[FilingMeta],
        skip_existing: bool = True,
    ) -> list[FilingMeta]:
        """
        Filing 목록의 HTML을 모두 다운로드.

        저장 구조:
          data/raw/html/{form_type}/{cik}_{accession}.html

        멱등성 보장:
          이미 다운로드된 파일은 skip (skip_existing=True)

        Args:
            filings:       FilingMeta 목록
            skip_existing: 이미 있는 파일 건너뛰기

        Returns:
            다운로드된 FilingMeta 목록 (downloaded=True)
        """
        results = []
        failed  = []

        for filing in tqdm(filings, desc="HTML 다운로드"):
            save_path = self._get_save_path(filing)

            # 이미 존재하면 skip
            if skip_existing and save_path.exists():
                filing.downloaded = True
                filing.file_size_kb = save_path.stat().st_size / 1024
                results.append(filing)
                continue

            # 다운로드 시도
            if not filing.html_url:
                logger.warning(f"URL 없음: {filing.company_name} ({filing.accession_no})")
                failed.append(filing)
                continue

            try:
                resp = self._get(filing.html_url)
                html = resp.text

                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_path.write_text(html, encoding="utf-8")

                filing.downloaded   = True
                filing.file_size_kb = save_path.stat().st_size / 1024
                results.append(filing)

                logger.debug(
                    f"저장: {filing.company_name} "
                    f"({filing.form_type}) "
                    f"{filing.file_size_kb:.1f}KB"
                )

            except Exception as e:
                logger.error(f"다운로드 실패: {filing.company_name}: {e}")
                failed.append(filing)

        logger.info(
            f"다운로드 완료: {len(results)}건 성공 / "
            f"{len(failed)}건 실패"
        )
        return results

    def _get_save_path(self, filing: FilingMeta) -> Path:
        """
        Filing의 로컬 저장 경로 결정.

        구조: data/raw/html/{form_type}/{cik}_{acc_hash}.html
        """
        acc_hash = hashlib.md5(filing.accession_no.encode()).hexdigest()[:8]
        fname    = f"{filing.cik}_{acc_hash}.html"
        return self.raw_dir / filing.form_type / fname

    # ── 메타데이터 저장/로드 ─────────────────────────────
    def save_manifest(
        self,
        filings:  list[FilingMeta],
        out_path: Path | None = None,
    ) -> Path:
        """
        수집된 Filing 메타데이터를 JSON으로 저장.

        이후 단계(파싱, 추출)에서 manifest를 읽어 처리.
        """
        if out_path is None:
            out_path = ROOT / self.cfg.paths.interim / "filing_manifest.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([asdict(fi) for fi in filings], f, indent=2)
        logger.info(f"Manifest 저장: {out_path} ({len(filings)}건)")
        return out_path

    @staticmethod
    def load_manifest(path: Path) -> list[FilingMeta]:
        """JSON manifest → FilingMeta 목록 복원"""
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [FilingMeta(**item) for item in raw]
