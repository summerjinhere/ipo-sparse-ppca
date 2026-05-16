"""
features/numerical_extractor.py
────────────────────────────────
424B4 HTML에서 수치 필드를 Rule-based로 추출

추출 전략:
  정형화 High  → 정규식 직접 추출
  정형화 Med   → 섹션 탐지 후 정규식
  정형화 Low   → 섹션 탐지 후 LLM (별도 모듈)

추출 필드 (Section 8 설계 기준):
  offer_price, filing_range_low/high,
  public_float_shares, lock_up_days,
  saleable_shares_180d, underwriter_count

주의: raw HTML 직접 투입 금지 (Polak & Morgan 2024)
      → 섹션 탐지 후 클린 텍스트 변환 선행 필수
"""

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from bs4 import BeautifulSoup
from loguru import logger


# ── 추출 결과 데이터 클래스 ────────────────────────────
@dataclass
class NumericalFields:
    """
    424B4에서 Rule-based로 추출된 수치 필드.

    필드별 타입 정의:
      float: 연속 수치
      int:   이산 수치
      str:   식별자
      None:  추출 실패

    downstream 처리:
      → PostgreSQL 저장 (src/models/database.py)
      → unlock_pressure 등 파생 변수 계산
    """
    # 식별자
    cik:              str   = ""
    company_name:     str   = ""
    filed_date:       str   = ""
    accession_no:     str   = ""

    # 공모 수치 (High 정형화 → 직접 추출)
    offer_price:           Optional[float] = None
    filing_range_low:      Optional[float] = None
    filing_range_high:     Optional[float] = None
    shares_offered:        Optional[float] = None   # 공모 주식 수
    public_float_shares:   Optional[float] = None   # 상장 후 유통 주식

    # 락업 (Medium 정형화 → 섹션 탐지 후 추출)
    lock_up_days:          Optional[int]   = None
    saleable_shares_180d:  Optional[float] = None

    # 주관사 (High 정형화)
    underwriter_name:      Optional[str]   = None
    underwriter_count:     Optional[int]   = None

    # 파생 변수 (추출 후 계산)
    unlock_pressure:       Optional[float] = None   # saleable / float
    price_update_abs:      Optional[float] = None   # |offer - mid| / mid
    filing_range_width:    Optional[float] = None   # (high-low)/mid
    relative_offer_size:   Optional[float] = None   # float / total_shares
    days_s1_to_424b4:      Optional[int]   = None
    amendment_count:       Optional[int]   = None

    # QC 메타데이터
    parse_success:    bool = True
    parse_errors:     list = field(default_factory=list)
    null_fields:      list = field(default_factory=list)

    def compute_derived(self) -> None:
        """
        수치 추출 후 파생 변수 계산.

        unlock_pressure:
          = saleable_shares_180d / public_float_shares
          Field & Hanka(2001)의 operationalization:
            "IPO 공모 주식 대비 180일 후 매도 가능 물량 비율"

        price_update_abs:
          = |offer_price - range_mid| / range_mid
          Hanley(1993) Partial Adjustment Phenomenon
          수요 불확실성의 프록시
        """
        errors = []

        # unlock_pressure
        if self.saleable_shares_180d and self.public_float_shares:
            if self.public_float_shares > 0:
                self.unlock_pressure = (
                    self.saleable_shares_180d / self.public_float_shares
                )
            else:
                errors.append("public_float_shares = 0, unlock_pressure 계산 불가")
        else:
            self.null_fields.append("unlock_pressure")

        # price_update_abs
        if self.offer_price and self.filing_range_low and self.filing_range_high:
            range_mid = (self.filing_range_low + self.filing_range_high) / 2
            if range_mid > 0:
                self.price_update_abs = (
                    abs(self.offer_price - range_mid) / range_mid
                )
        else:
            self.null_fields.append("price_update_abs")

        # filing_range_width
        if self.filing_range_low and self.filing_range_high:
            range_mid = (self.filing_range_low + self.filing_range_high) / 2
            if range_mid > 0:
                self.filing_range_width = (
                    (self.filing_range_high - self.filing_range_low) / range_mid
                )

        if errors:
            self.parse_errors.extend(errors)

    def qc_check(self) -> bool:
        """
        핵심 필드 null rate 확인.

        핵심 필드 중 하나라도 None이면 QC 실패.
        이후 LLM 교차검증 또는 수동 검토 대상으로 플래그.

        Returns:
            True → QC 통과, False → 검토 필요
        """
        critical = [
            "offer_price",
            "public_float_shares",
            "lock_up_days",
            "saleable_shares_180d",
        ]
        missing = [f for f in critical if getattr(self, f) is None]
        if missing:
            self.parse_success = False
            self.null_fields.extend(missing)
            return False
        return True


# ── 섹션 탐지기 ────────────────────────────────────────
class SectionDetector:
    """
    424B4 HTML에서 특정 섹션을 탐지하고 클린 텍스트 반환.

    왜 섹션 탐지가 선행되어야 하는가:
      - Raw HTML에 LLM 직접 적용 시 F1 ~ 13-14% (Polak & Morgan 2024)
      - 섹션 분리 후 클린 텍스트 제공 시 정확도 대폭 향상
      - 불필요한 HTML 태그, 스크립트, 헤더 제거

    탐지 전략:
      1. 표준 CSS class 기반 ("Section", "lockup" 등)
      2. 헤더 텍스트 매칭 ("Lock-Up Agreements" 등)
      3. 페이지 내 순서 기반 (fallback)
    """

    # 섹션별 탐지 패턴 정의
    SECTION_PATTERNS = {
        "lockup": [
            r"lock[\s\-]?up\s+agreement",
            r"lock[\s\-]?up\s+arrangement",
            r"shares\s+eligible\s+for\s+future\s+sale",
            r"selling\s+restriction",
            r"transfer\s+restriction",
        ],
        "pricing": [
            r"price\s+of\s+the\s+offering",
            r"offering\s+price",
            r"public\s+offering\s+price",
        ],
        "underwriters": [
            r"underwriting",
            r"underwriter",
        ],
        "capitalization": [
            r"capitalization",
            r"shares\s+outstanding",
        ],
    }

    def __init__(self, html: str):
        self.soup = BeautifulSoup(html, "lxml")
        # 스크립트, 스타일, 헤더 제거
        for tag in self.soup(["script", "style", "header", "footer", "nav"]):
            tag.decompose()

    def find_section(self, section_name: str) -> str:
        """
        특정 섹션의 클린 텍스트 반환.

        탐지 순서:
          1. CSS class 기반
          2. 헤더 텍스트 매칭
          3. 패턴 매칭 (fallback)

        Args:
            section_name: "lockup" / "pricing" / "underwriters" / "capitalization"

        Returns:
            클린 텍스트 (섹션을 찾지 못하면 "")
        """
        patterns = self.SECTION_PATTERNS.get(section_name, [])

        # 전략 1: 헤더 텍스트 매칭
        for tag in self.soup.find_all(["h1", "h2", "h3", "h4", "b", "strong"]):
            tag_text = tag.get_text(" ", strip=True).lower()
            for pattern in patterns:
                if re.search(pattern, tag_text, re.IGNORECASE):
                    return self._extract_after_header(tag)

        # 전략 2: 전체 텍스트 패턴 매칭 (fallback)
        full_text = self.soup.get_text(" ", strip=True)
        for pattern in patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                start = max(0, match.start() - 100)
                end   = min(len(full_text), match.end() + 3000)
                return full_text[start:end]

        logger.debug(f"섹션 탐지 실패: {section_name}")
        return ""

    def _extract_after_header(self, header_tag, max_chars: int = 3000) -> str:
        """
        헤더 태그 이후 텍스트 추출.

        헤더 다음 형제(sibling) 및 자식 노드에서
        max_chars만큼 텍스트를 수집.
        """
        texts = []
        char_count = 0
        for sibling in header_tag.find_next_siblings():
            text = sibling.get_text(" ", strip=True)
            texts.append(text)
            char_count += len(text)
            if char_count > max_chars:
                break
        return " ".join(texts)


# ── 수치 추출기 ────────────────────────────────────────
class NumericalExtractor:
    """
    섹션 클린 텍스트에서 수치 필드를 정규식으로 추출.

    정형화 수준별 전략:
      High (offer_price, lock_up_days):
        → 고정 패턴, 높은 신뢰도
      Medium (public_float, saleable_shares):
        → 섹션 탐지 선행 후 정규식

    사용법:
        html = Path("data/raw/html/424B4/0001234.html").read_text()
        extractor = NumericalExtractor(html, meta)
        fields = extractor.extract()
        fields.compute_derived()
        fields.qc_check()
    """

    def __init__(self, html: str, meta: dict):
        self.detector = SectionDetector(html)
        self.full_text = self.detector.soup.get_text(" ", strip=True)
        self.meta = meta

    def extract(self) -> NumericalFields:
        """전체 수치 필드 추출 실행"""
        fields = NumericalFields(
            cik=self.meta.get("cik", ""),
            company_name=self.meta.get("company_name", ""),
            filed_date=self.meta.get("filed_date", ""),
            accession_no=self.meta.get("accession_no", ""),
        )

        # 섹션별 클린 텍스트 추출
        pricing_text  = self.detector.find_section("pricing")
        lockup_text   = self.detector.find_section("lockup")
        uw_text       = self.detector.find_section("underwriters")
        cap_text      = self.detector.find_section("capitalization")

        # 각 필드 추출
        fields.offer_price          = self._extract_offer_price(pricing_text)
        fields.filing_range_low, \
        fields.filing_range_high    = self._extract_filing_range(self.full_text)
        fields.lock_up_days         = self._extract_lockup_days(lockup_text)
        fields.saleable_shares_180d = self._extract_saleable_shares(lockup_text)
        fields.public_float_shares  = self._extract_public_float(cap_text)
        fields.underwriter_name, \
        fields.underwriter_count    = self._extract_underwriters(uw_text)

        return fields

    # ── 개별 필드 추출 메서드 ────────────────────────────

    def _extract_offer_price(self, text: str) -> Optional[float]:
        """
        공모가 추출. 정형화: High.

        패턴 예시 (실제 424B4):
          "The public offering price is $78.00 per share"
          "Price per share: $68.00"
          "at a price of $18.00 per share"
        """
        patterns = [
            r"price\s+(?:per\s+share|to\s+the\s+public)\s+(?:is\s+)?\$\s*([\d,]+\.?\d*)",
            r"public\s+offering\s+price\s+(?:of\s+)?\$\s*([\d,]+\.?\d*)",
            r"at\s+a\s+price\s+of\s+\$\s*([\d,]+\.?\d*)",
            r"priced\s+at\s+\$\s*([\d,]+\.?\d*)",
        ]
        return self._match_price(text, patterns)

    def _extract_filing_range(self, text: str) -> tuple[Optional[float], Optional[float]]:
        """
        Filing range (low, high) 추출. 정형화: High.

        패턴 예시:
          "between $72.00 and $74.00 per share"
          "estimated price range of $16.00 to $18.00"
        """
        patterns = [
            r"between\s+\$\s*([\d,]+\.?\d*)\s+and\s+\$\s*([\d,]+\.?\d*)",
            r"price\s+range\s+of\s+\$\s*([\d,]+\.?\d*)\s+to\s+\$\s*([\d,]+\.?\d*)",
            r"\$\s*([\d,]+\.?\d*)\s+to\s+\$\s*([\d,]+\.?\d*)\s+per\s+share",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                try:
                    low  = float(m.group(1).replace(",", ""))
                    high = float(m.group(2).replace(",", ""))
                    return low, high
                except (ValueError, IndexError):
                    continue
        return None, None

    def _extract_lockup_days(self, text: str) -> Optional[int]:
        """
        락업 기간(일) 추출. 정형화: Medium.

        패턴 예시:
          "subject to a 180-day lock-up period"
          "lock-up period of 180 days"
          "for a period of 180 days after"
        """
        patterns = [
            r"(\d+)[\s\-]day\s+lock[\s\-]?up",
            r"lock[\s\-]?up\s+period\s+of\s+(\d+)\s+days",
            r"for\s+a\s+period\s+of\s+(\d+)\s+days",
            r"(\d+)\s+days?\s+(?:after|from)\s+the\s+date\s+of\s+this\s+prospectus",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                try:
                    days = int(m.group(1))
                    # 합리적 범위 검증 (30일 ~ 730일)
                    if 30 <= days <= 730:
                        return days
                except (ValueError, IndexError):
                    continue
        return None

    def _extract_saleable_shares(self, text: str) -> Optional[float]:
        """
        180일 락업 해제 후 매도 가능 주식 수. 정형화: Medium.

        주의:
          - 숫자 단위가 혼재 (shares / million shares)
          - "approximately" 등 수식어 처리
          - 복수 락업 그룹 합산 필요 (복잡 케이스)

        단순화:
          첫 번째로 매칭되는 대규모 주식 수를 사용
          복잡 케이스는 LLM 어노테이션으로 플래그
        """
        patterns = [
            r"([\d,]+)\s+shares\s+(?:will\s+be\s+)?(?:become|eligible|available)\s+for\s+sale",
            r"([\d,]+)\s+shares?\s+subject\s+to\s+(?:the\s+)?lock[\s\-]?up",
            r"lock[\s\-]?up\s+covers?\s+([\d,]+)\s+shares?",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                try:
                    val = float(m.group(1).replace(",", ""))
                    if val > 1000:   # 최소 1,000주 이상
                        return val
                except (ValueError, IndexError):
                    continue
        return None

    def _extract_public_float(self, text: str) -> Optional[float]:
        """
        IPO 후 유통 주식 수 (public float). 정형화: Medium.
        """
        patterns = [
            r"([\d,]+)\s+shares?\s+(?:will\s+be\s+)?outstanding\s+after\s+this\s+offering",
            r"([\d,]+)\s+shares?\s+of\s+(?:our\s+)?class\s+[ab]\s+common\s+stock\s+outstanding",
            r"total\s+of\s+([\d,]+)\s+shares?\s+outstanding",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                try:
                    val = float(m.group(1).replace(",", ""))
                    if val > 10_000:   # 최소 10,000주
                        return val
                except (ValueError, IndexError):
                    continue
        return None

    def _extract_underwriters(self, text: str) -> tuple[Optional[str], Optional[int]]:
        """
        주관사 이름 및 수 추출. 정형화: High.

        알려진 주요 주관사 목록으로 매칭:
          Goldman Sachs, Morgan Stanley, JP Morgan, ...
        """
        known_underwriters = [
            "Goldman Sachs", "Morgan Stanley", "J.P. Morgan",
            "JPMorgan", "Citigroup", "BofA", "Bank of America",
            "Credit Suisse", "Deutsche Bank", "Barclays",
            "Wells Fargo", "UBS", "Jefferies", "Cowen",
            "Piper Sandler", "William Blair",
        ]
        found = []
        for uw in known_underwriters:
            if uw.lower() in text.lower():
                found.append(uw)

        if found:
            return found[0], len(found)
        return None, None

    # ── 공통 헬퍼 ───────────────────────────────────────
    def _match_price(
        self,
        text:     str,
        patterns: list[str],
    ) -> Optional[float]:
        """패턴 목록 순서대로 시도 → 첫 매칭 반환"""
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1).replace(",", ""))
                except (ValueError, IndexError):
                    continue
        return None
