"""
evaluation/span_test.py
────────────────────────
Span Test 및 CAR 기반 평가

핵심 검정:
  H₀: γ = 0  (추출된 팩터가 CAR을 설명 못함)
  H₁: γ ≠ 0  (팩터가 CAR을 span함)

  → 기각이면: "오버행 특성이 span하는 팩터가
               락업 해제 비정상 수익률을 설명"

Risk vs Anomaly 구별:
  IPCA 프레임워크의 핵심 기여
  α (intercept) 검정으로 구별
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy import stats
from loguru import logger


# ── 검정 결과 ─────────────────────────────────────────
@dataclass
class SpanTestResult:
    """
    Span Test 결과.

    해석:
      rejected = True → 팩터가 CAR을 span함
                        오버행 특성이 수익률 예측에 유의미
      rejected = False → span 못함
    """
    # 검정 통계량
    F_stat:   float = 0.0
    p_value:  float = 1.0
    rejected: bool  = False   # H₀ 기각 여부

    # 계수 추정
    gamma:    np.ndarray = field(default_factory=lambda: np.array([]))
    gamma_se: np.ndarray = field(default_factory=lambda: np.array([]))
    t_stats:  np.ndarray = field(default_factory=lambda: np.array([]))

    # R²
    r2:       float = 0.0

    # Bootstrap (선택)
    boot_p_value: float = 1.0
    n_boot:       int   = 0

    # Risk vs Anomaly
    alpha:     float = 0.0    # intercept (≠0 이면 anomaly 성분 존재)
    alpha_se:  float = 0.0
    alpha_t:   float = 0.0

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "Span Test 결과",
            "=" * 50,
            f"F-통계량:     {self.F_stat:.4f}",
            f"p-value:      {self.p_value:.4f}",
            f"H₀ 기각:      {'✅ 예' if self.rejected else '❌ 아니오'}",
            f"R²:           {self.r2:.4f}",
            f"",
            f"Risk vs Anomaly 구별:",
            f"  α (intercept): {self.alpha:.4f} (t={self.alpha_t:.3f})",
            f"  α=0 이면 → Risk-based 설명",
            f"  α≠0 이면 → Anomaly 성분 존재",
            "=" * 50,
        ]
        return "\n".join(lines)


# ── 메인 평가 클래스 ───────────────────────────────────
class SpanTest:
    """
    Sparse PPCA 팩터 → CAR Span Test.

    검정 방정식:
      CARᵢ = α + Σₖ γₖ·F̂ᵢₖ + εᵢ

      여기서 F̂ᵢ = 기업 i의 팩터 스코어 (k × 1)

    Wald Test:
      H₀: γ₁ = γ₂ = ... = γₖ = 0
      통계량: W = γ̂'·V⁻¹·γ̂  ~ χ²(k)

    사용법:
        tester = SpanTest(significance_level=0.05)
        result = tester.test(
            car=car_array,        # (N,) CAR 벡터
            F_scores=F_hat,       # (N, k) 팩터 스코어
            boot_reps=1000,
        )
        print(result.summary())
    """

    def __init__(self, significance_level: float = 0.05):
        self.alpha_level = significance_level

    def compute_car(
        self,
        returns:      np.ndarray,   # (N × T) 이벤트 타임 수익률
        event_col:    int   = 0,    # t=0 기준 열 인덱스
        window:       tuple = (-1, 1),
        market_ret:   np.ndarray | None = None,  # (T,) 시장 수익률
    ) -> np.ndarray:
        """
        누적 비정상 수익률(CAR) 계산.

        방법:
          시장 조정 수익률 = Rᵢ,ₜ - Rₘ,ₜ
          CAR[-1,+1] = Σₜ∈[-1,+1] (Rᵢ,ₜ - Rₘ,ₜ)

        Args:
            returns:    (N × T) 이벤트 타임 수익률
            event_col:  이벤트 발생 열 인덱스 (t=0)
            window:     (-1, 1) → 전일 ~ 익일
            market_ret: (T,) 시장 수익률 (없으면 Market Model 생략)

        Returns:
            car: (N,) 기업별 CAR
        """
        N, T = returns.shape
        t_start, t_end = window

        # 윈도우 내 열 인덱스
        cols = range(
            max(0, event_col + t_start),
            min(T, event_col + t_end + 1)
        )

        if len(cols) == 0:
            logger.warning("CAR 계산 윈도우가 데이터 범위를 벗어남")
            return np.zeros(N)

        # 비정상 수익률
        if market_ret is not None:
            ab_ret = returns[:, list(cols)] - market_ret[list(cols)]
        else:
            # 단순 평균 차감 (Market Model 없을 때)
            ab_ret = returns[:, list(cols)] - returns[:, list(cols)].mean(axis=0)

        car = ab_ret.sum(axis=1)
        logger.info(
            f"CAR 계산 완료: window={window}, "
            f"mean={car.mean():.4f}, std={car.std():.4f}"
        )
        return car

    def test(
        self,
        car:        np.ndarray,   # (N,)
        F_scores:   np.ndarray,   # (N, k)
        boot_reps:  int  = 1000,
        include_alpha: bool = True,
    ) -> SpanTestResult:
        """
        Span Test 실행.

        Step 1: OLS 회귀
          CARᵢ = α + Σₖ γₖ·F̂ᵢₖ + εᵢ

        Step 2: Wald Test (H₀: γ = 0)
          W = γ̂'·(Var[γ̂])⁻¹·γ̂ ~ χ²(k)

        Step 3: Bootstrap p-value
          교란된 데이터에서 W 분포 추정

        Args:
            car:          (N,) CAR 벡터
            F_scores:     (N, k) 팩터 스코어
            boot_reps:    Bootstrap 반복 횟수
            include_alpha: 절편 포함 여부

        Returns:
            SpanTestResult
        """
        N, k = F_scores.shape
        result = SpanTestResult()

        # Step 1: OLS
        if include_alpha:
            X = np.column_stack([np.ones(N), F_scores])   # (N, 1+k)
        else:
            X = F_scores   # (N, k)

        # OLS: γ̂ = (X'X)⁻¹ X'y
        XtX = X.T @ X
        Xty = X.T @ car
        try:
            coef = np.linalg.solve(XtX + 1e-8 * np.eye(X.shape[1]), Xty)
        except np.linalg.LinAlgError:
            logger.error("OLS 풀이 실패: X'X가 특이행렬")
            return result

        # 잔차 및 분산
        resid  = car - X @ coef
        sigma2 = np.sum(resid ** 2) / (N - X.shape[1])
        cov    = sigma2 * np.linalg.inv(XtX + 1e-8 * np.eye(X.shape[1]))

        if include_alpha:
            alpha_hat  = coef[0]
            gamma_hat  = coef[1:]
            gamma_cov  = cov[1:, 1:]
            alpha_var  = cov[0, 0]
        else:
            alpha_hat  = 0.0
            gamma_hat  = coef
            gamma_cov  = cov
            alpha_var  = 0.0

        # 표준오차 및 t-통계량
        gamma_se = np.sqrt(np.diag(gamma_cov))
        t_stats  = gamma_hat / (gamma_se + 1e-10)

        # R²
        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum((car - car.mean()) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-10)

        # Step 2: Wald Test
        try:
            W = gamma_hat @ np.linalg.solve(
                gamma_cov + 1e-8 * np.eye(k),
                gamma_hat
            )
        except np.linalg.LinAlgError:
            W = 0.0

        p_val    = 1.0 - stats.chi2.cdf(W, df=k)
        rejected = p_val < self.alpha_level

        # α t-통계량 (Risk vs Anomaly)
        alpha_t = alpha_hat / (np.sqrt(alpha_var) + 1e-10)

        result.F_stat   = float(W)
        result.p_value  = float(p_val)
        result.rejected = rejected
        result.gamma    = gamma_hat
        result.gamma_se = gamma_se
        result.t_stats  = t_stats
        result.r2       = float(max(0.0, r2))
        result.alpha    = float(alpha_hat)
        result.alpha_se = float(np.sqrt(alpha_var))
        result.alpha_t  = float(alpha_t)

        # Step 3: Bootstrap
        if boot_reps > 0:
            result.boot_p_value = self._bootstrap_pval(
                car, F_scores, W, boot_reps, include_alpha
            )
            result.n_boot = boot_reps

        logger.info(result.summary())
        return result

    def _bootstrap_pval(
        self,
        car:       np.ndarray,
        F_scores:  np.ndarray,
        W_obs:     float,
        n_boot:    int,
        include_alpha: bool,
    ) -> float:
        """
        Bootstrap p-value 계산.

        아이디어:
          H₀ 하에서 Wald 통계량 W의 분포를 
          잔차 permutation으로 추정.

          잔차를 랜덤으로 섞으면 CAR와 F 사이의
          관계가 끊어짐 → H₀ 상황 모사.
        """
        N = len(car)
        rng = np.random.default_rng(42)
        W_boot = np.zeros(n_boot)

        # null 잔차 생성 (H₀ 하 잔차 = H₀ 기각 후 OLS 잔차)
        if include_alpha:
            X = np.column_stack([np.ones(N), F_scores])
        else:
            X = F_scores
        coef_null = np.linalg.lstsq(X, car, rcond=None)[0]
        resid_null = car - X @ coef_null

        for b in range(n_boot):
            # 잔차 shuffle
            car_b = resid_null[rng.permutation(N)]

            # Wald 통계량 재계산
            Xty = X.T @ car_b
            try:
                coef_b = np.linalg.solve(X.T @ X + 1e-8 * np.eye(X.shape[1]), Xty)
                if include_alpha:
                    g = coef_b[1:]
                    cov_g = (np.sum(car_b - X@coef_b)**2 / (N - X.shape[1])
                             ) * np.linalg.inv(X.T @ X + 1e-8 * np.eye(X.shape[1]))[1:, 1:]
                else:
                    g = coef_b
                    cov_g = (np.sum(car_b - X@coef_b)**2 / (N - X.shape[1])
                             ) * np.linalg.inv(X.T @ X + 1e-8 * np.eye(X.shape[1]))
                W_boot[b] = float(g @ np.linalg.solve(cov_g + 1e-8*np.eye(len(g)), g))
            except Exception:
                W_boot[b] = 0.0

        boot_pval = np.mean(W_boot >= W_obs)
        return float(boot_pval)
