"""
models/sparse_ppca.py
─────────────────────
Sparse Projected PCA 핵심 구현

이론 기반:
  Fan, Liao & Wang (2016) Annals of Statistics
    → Projected PCA 원전
  Kelly, Pruitt & Su (2019) JFE
    → IPCA (PPCA의 쌍둥이 프레임워크)
  Bybee et al. (Narrative Asset Pricing)
    → Sparse IPCA (Group LASSO 확장)

알고리즘:
  교대 최소화 (Alternating Minimization)
    Step A: F 업데이트 (Γ 고정, OLS)
    Step B: Γ 업데이트 (F 고정, Group Soft-Thresholding)
    Step C: 수렴 확인

수렴 보장:
  매 반복마다 목적함수 단조 감소
  (각 Step이 볼록 부분 문제의 정확한 최솟값)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger


# ── 추정 결과 ──────────────────────────────────────────
@dataclass
class SparsePPCAResult:
    """
    Sparse PPCA 추정 결과.

    핵심 출력:
      Gamma: 특성 → 팩터 로딩 매핑 (p × k)
             0인 행 = 선택되지 않은 특성
             0이 아닌 행 = 선택된 특성

      F:     잠재 팩터 시계열 (T × k)
             각 열 = 하나의 IPO 오버행 팩터

      selected_features: 선택된 특성 인덱스 목록
    """
    # 핵심 파라미터
    Gamma:    np.ndarray  = field(default_factory=lambda: np.array([]))  # p × k
    F:        np.ndarray  = field(default_factory=lambda: np.array([]))  # T × k
    Lambda:   np.ndarray  = field(default_factory=lambda: np.array([]))  # N × k  (= Z @ Gamma)

    # 수렴 정보
    n_iter:       int   = 0
    converged:    bool  = False
    obj_history:  list  = field(default_factory=list)

    # Sparse 결과
    lambda_opt:        float = 0.0
    selected_features: list  = field(default_factory=list)   # 선택된 특성 인덱스
    n_selected:        int   = 0

    # 적합도
    r2_total:    float = 0.0   # 전체 설명력
    r2_per_factor: list = field(default_factory=list)

    def summary(self) -> str:
        """추정 결과 요약 출력"""
        lines = [
            "=" * 50,
            "Sparse PPCA 추정 결과",
            "=" * 50,
            f"팩터 수 (k):          {self.F.shape[1] if self.F.size else 0}",
            f"선택된 특성 수:        {self.n_selected} / {self.Gamma.shape[0] if self.Gamma.size else 0}",
            f"선택된 특성 인덱스:    {self.selected_features}",
            f"최적 λ:               {self.lambda_opt:.6f}",
            f"수렴 여부:            {'✅' if self.converged else '❌'}",
            f"반복 횟수:            {self.n_iter}",
            f"총 R²:                {self.r2_total:.4f}",
            "=" * 50,
        ]
        return "\n".join(lines)


# ── 핵심 알고리즘 ──────────────────────────────────────
class SparsePPCA:
    """
    Sparse Projected PCA 구현.

    수학적 구조:
      R:    수익률 행렬 (N × T)
            N = IPO 기업 수, T = 이벤트 타임
      Z:    특성 행렬 (N × p)
            p = 특성 수 (Level A + B + C)

    목적 함수:
      min_{Γ, F}  ‖R̃ - ZΓF'‖²_F  +  λΣⱼ‖Γⱼ‖₂

      R̃ = P_Z · R  (Z에 사영된 수익률)
      P_Z = Z(Z'Z)⁻¹Z'

    사용법:
        model = SparsePPCA(k=3, lambda_grid_size=50)
        result = model.fit(R, Z)
        print(result.summary())

        # 선택된 특성 확인
        print(result.selected_features)

        # Span Test용 팩터 스코어
        F_hat = result.F   # (T × k)
    """

    def __init__(
        self,
        k:                int   = 3,
        max_iter:         int   = 500,
        tol:              float = 1e-6,
        n_init:           int   = 10,
        lambda_grid_size: int   = 50,
        lambda_min_ratio: float = 1e-4,
        seed:             int   = 42,
    ):
        """
        Args:
            k:                팩터 수 (IC 기준으로 결정하거나 직접 지정)
            max_iter:         최대 반복 횟수
            tol:              수렴 임계값 (목적함수 상대 변화량)
            n_init:           다중 초기화 횟수 (non-convex 문제 대응)
            lambda_grid_size: λ 탐색 그리드 크기
            lambda_min_ratio: λ_min = λ_max × lambda_min_ratio
            seed:             재현성 확보
        """
        self.k                = k
        self.max_iter         = max_iter
        self.tol              = tol
        self.n_init           = n_init
        self.lambda_grid_size = lambda_grid_size
        self.lambda_min_ratio = lambda_min_ratio
        self.rng              = np.random.default_rng(seed)

    # ─────────────────────────────────────────────────────
    # 메인 인터페이스
    # ─────────────────────────────────────────────────────

    def fit(
        self,
        R:      np.ndarray,   # (N × T) 수익률 행렬
        Z:      np.ndarray,   # (N × p) 특성 행렬
        lambda_val: Optional[float] = None,
    ) -> SparsePPCAResult:
        """
        Sparse PPCA 추정.

        lambda_val 지정 시:
          해당 λ로 단일 추정

        lambda_val = None:
          Cross-Validation으로 최적 λ 자동 선택

        Args:
            R:          수익률 행렬 (N × T)
            Z:          특성 행렬 (N × p), 표준화 권장
            lambda_val: 규제 강도 (None이면 CV로 선택)

        Returns:
            SparsePPCAResult
        """
        N, T = R.shape
        _, p = Z.shape
        logger.info(f"Sparse PPCA 시작: N={N}, T={T}, p={p}, k={self.k}")

        # Step 1: Z 표준화 (이미 되어있으면 skip)
        Z_std = self._standardize(Z)

        # Step 2: 사영 행렬 + 사영된 수익률
        P_Z, R_tilde = self._project(R, Z_std)
        logger.debug(f"사영 완료: R̃ shape = {R_tilde.shape}")

        # Step 3: λ 결정
        if lambda_val is None:
            lambda_val = self._select_lambda_cv(R_tilde, Z_std, R)
            logger.info(f"CV 최적 λ: {lambda_val:.6f}")
        else:
            logger.info(f"지정된 λ: {lambda_val:.6f}")

        # Step 4: 최적 λ로 최종 추정
        Gamma, F, n_iter, converged, obj_hist = self._fit_single(
            R_tilde, Z_std, lambda_val
        )

        # Step 5: 결과 조립
        result = self._build_result(
            Gamma, F, Z_std, R_tilde,
            n_iter, converged, obj_hist, lambda_val
        )
        logger.info(result.summary())
        return result

    # ─────────────────────────────────────────────────────
    # 사영 (Projection)
    # ─────────────────────────────────────────────────────

    def _project(
        self,
        R: np.ndarray,
        Z: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        수익률 R을 특성 공간 Z에 사영.

        P_Z = Z(Z'Z)⁻¹Z'  (N × N 사영 행렬)
        R̃  = P_Z · R       (N × T 사영된 수익률)

        직관:
          R̃_{i,t} = R_{i,t}의 Z_i로 설명되는 성분
          Z와 무관한 노이즈 제거됨

        수치 안정성:
          (Z'Z)의 역행렬 대신 lstsq 사용
          → 특성 간 다중공선성 대응
        """
        N, p = Z.shape

        # Z'Z의 역행렬 → lstsq로 대체 (수치 안정)
        # P_Z @ R = Z @ (Z'Z)⁻¹ @ Z' @ R
        # = Z @ lstsq(Z, R)

        # Z'Z가 특이(singular)할 경우 Ridge regularization 추가
        ZtZ = Z.T @ Z
        ridge = 1e-8 * np.trace(ZtZ) / p
        ZtZ_reg = ZtZ + ridge * np.eye(p)

        # P_Z @ R = Z @ inv(Z'Z) @ Z' @ R
        ZtR    = Z.T @ R                          # (p × T)
        coef   = np.linalg.solve(ZtZ_reg, ZtR)   # (p × T)
        R_tilde = Z @ coef                         # (N × T)

        # P_Z 자체는 저장하지 않음 (N×N → 메모리 비효율)
        # 필요 시 재계산
        P_Z = None

        return P_Z, R_tilde

    # ─────────────────────────────────────────────────────
    # 교대 최소화 알고리즘
    # ─────────────────────────────────────────────────────

    def _fit_single(
        self,
        R_tilde:    np.ndarray,   # (N × T)
        Z:          np.ndarray,   # (N × p)
        lambda_val: float,
    ) -> tuple[np.ndarray, np.ndarray, int, bool, list[float]]:
        """
        단일 λ에 대한 Sparse PPCA 교대 최소화.

        다중 초기화 → 목적함수 최소인 해 선택.

        Step A: F 업데이트 (Γ 고정)
          F̂ = R̃'·(ZΓ)·[(ZΓ)'(ZΓ)]⁻¹

        Step B: Γ 업데이트 (F 고정)
          각 j에 Group Soft-Thresholding 적용

        Step C: 수렴 확인
          |f_new - f_old| / (|f_old| + 1e-10) < tol

        Returns:
            (Gamma, F, n_iter, converged, obj_history)
        """
        N, T = R_tilde.shape
        _, p = Z.shape
        k    = self.k

        best_obj   = np.inf
        best_Gamma = None
        best_F     = None
        best_n_iter = 0
        best_conv  = False
        best_hist  = []

        for init_idx in range(self.n_init):
            # 초기화: PCA 기반 (첫 번째) 또는 랜덤
            if init_idx == 0:
                Gamma, F = self._init_from_pca(R_tilde, Z)
            else:
                Gamma = self.rng.standard_normal((p, k))
                F     = self.rng.standard_normal((T, k))

            # 교대 최소화
            obj_history = []
            converged   = False

            for it in range(self.max_iter):
                # Step A: F 업데이트
                F = self._update_F(R_tilde, Z, Gamma)

                # Step B: Γ 업데이트 (Group Soft-Thresholding)
                Gamma = self._update_Gamma(R_tilde, Z, F, lambda_val)

                # 목적함수 계산
                obj = self._objective(R_tilde, Z, Gamma, F, lambda_val)
                obj_history.append(obj)

                # 수렴 확인
                if it > 0:
                    rel_change = abs(obj_history[-2] - obj) / (abs(obj_history[-2]) + 1e-10)
                    if rel_change < self.tol:
                        converged = True
                        logger.debug(f"[init={init_idx}] 수렴: {it+1}회 반복")
                        break

            # 최선 해 업데이트
            if obj_history[-1] < best_obj:
                best_obj    = obj_history[-1]
                best_Gamma  = Gamma.copy()
                best_F      = F.copy()
                best_n_iter = it + 1
                best_conv   = converged
                best_hist   = obj_history

        return best_Gamma, best_F, best_n_iter, best_conv, best_hist

    def _update_F(
        self,
        R_tilde: np.ndarray,   # (N × T)
        Z:       np.ndarray,   # (N × p)
        Gamma:   np.ndarray,   # (p × k)
    ) -> np.ndarray:           # (T × k)
        """
        Step A: Γ 고정, F 최적화.

        min_F  ‖R̃ - ZΓF'‖²_F
        → OLS 해: F = R̃'·A·(A'A)⁻¹
          A = ZΓ (N × k)

        직관:
          ZΓ를 "팩터 기저"로 놓고,
          R̃의 각 시점을 이 기저에 OLS 회귀

        수치 안정성:
          A'A 역행렬 → lstsq 사용
        """
        A   = Z @ Gamma   # (N × k)
        AtA = A.T @ A     # (k × k)
        AtR = A.T @ R_tilde  # (k × T)

        # Ridge regularization for stability
        ridge = 1e-8 * np.trace(AtA) / self.k
        AtA_reg = AtA + ridge * np.eye(self.k)

        # F = (A'A)⁻¹ A' R̃ → F' = (A'A)⁻¹ (A' R̃)
        # F (T × k): 각 행 = 시점 t의 팩터값
        F = np.linalg.solve(AtA_reg, AtR).T   # (T × k)
        return F

    def _update_Gamma(
        self,
        R_tilde:    np.ndarray,   # (N × T)
        Z:          np.ndarray,   # (N × p)
        F:          np.ndarray,   # (T × k)
        lambda_val: float,
    ) -> np.ndarray:              # (p × k)
        """
        Step B: F 고정, Γ 최적화 (Group Soft-Thresholding).

        min_Γ  ‖R̃ - ZΓF'‖² + λΣⱼ‖Γⱼ‖₂

        각 j에 대해 부분 문제:
          min_{Γⱼ}  ‖Rⱼ - ZⱼΓⱼ'F'‖² + λ‖Γⱼ‖₂
          
          OLS 해: ũⱼ = (Zⱼ'Zⱼ)⁻¹ Zⱼ'Rⱼ (F'F)⁻¹ F
          Group ST: Γ̂ⱼ = max(‖ũⱼ‖₂ - λ/2, 0) · ũⱼ/‖ũⱼ‖₂

        직관:
          OLS 해 ũⱼ의 벡터 크기가 λ/2보다 작으면 →  Γⱼ = 0 (제거)
          크면 → 0쪽으로 λ/2만큼 축소 후 방향 보존

        주의:
          반드시 F'F도 ridge 처리 필요
        """
        N, p = Z.shape
        T, k = F.shape
        Gamma = np.zeros((p, k))

        # F'F 사전 계산
        FtF = F.T @ F   # (k × k)
        ridge = 1e-8 * np.trace(FtF) / k
        FtF_reg = FtF + ridge * np.eye(k)

        for j in range(p):
            Zj = Z[:, j:j+1]   # (N × 1)

            # 잔차 계산: 다른 모든 특성의 기여 제외
            # R_j = R̃ - Σ_{j'≠j} Z_{j'}Γ_{j'}'F'
            Gamma_without_j = Gamma.copy()
            Gamma_without_j[j, :] = 0.0
            R_j = R_tilde - Z @ Gamma_without_j @ F.T   # (N × T)

            # OLS 해 계산
            # ũⱼ = [(Zⱼ'Zⱼ)⁻¹ Zⱼ'] @ R_j @ [F(F'F)⁻¹]
            ZjtZj = float((Zj.T @ Zj).squeeze()) + 1e-10   # 스칼라
            ZjtRj = Zj.T @ R_j                  # (1 × T)

            # F(F'F)⁻¹ 방향 계산
            FtF_inv_F = np.linalg.solve(FtF_reg, F.T)   # (k × T)
            u_j = (ZjtRj @ FtF_inv_F.T) / ZjtZj         # (1 × k) → (k,)
            u_j = u_j.flatten()

            # Group Soft-Thresholding
            norm_uj = np.linalg.norm(u_j)
            threshold = lambda_val / 2.0

            if norm_uj > threshold:
                # 축소: max(norm - threshold, 0) / norm × 방향
                scale = (norm_uj - threshold) / norm_uj
                Gamma[j, :] = scale * u_j
            else:
                # 제거: Γⱼ = 0 (이미 0으로 초기화됨)
                Gamma[j, :] = 0.0

        return Gamma

    # ─────────────────────────────────────────────────────
    # 목적 함수
    # ─────────────────────────────────────────────────────

    def _objective(
        self,
        R_tilde:    np.ndarray,
        Z:          np.ndarray,
        Gamma:      np.ndarray,
        F:          np.ndarray,
        lambda_val: float,
    ) -> float:
        """
        목적 함수값 계산.

        f(Γ, F) = ‖R̃ - ZΓF'‖²_F + λΣⱼ‖Γⱼ‖₂
        """
        residual = R_tilde - Z @ Gamma @ F.T
        fit_term     = np.sum(residual ** 2)
        penalty_term = lambda_val * sum(
            np.linalg.norm(Gamma[j, :])
            for j in range(Gamma.shape[0])
        )
        return fit_term + penalty_term

    # ─────────────────────────────────────────────────────
    # λ 선택 (Cross-Validation)
    # ─────────────────────────────────────────────────────

    def _select_lambda_cv(
        self,
        R_tilde: np.ndarray,
        Z:       np.ndarray,
        R_orig:  np.ndarray,
        n_folds: int = 5,
    ) -> float:
        """
        Out-of-Sample MSE 기반 λ 선택.

        λ 그리드:
          λ_max = 최대 OLS 해 벡터 norm × 2
          λ_min = λ_max × lambda_min_ratio
          등비수열로 lambda_grid_size개 생성

        CV 절차:
          IPO 기업(N 차원)을 n_folds로 분할
          각 fold → test set으로 OOS MSE 계산
          MSE 최소 λ 선택

        주의:
          시계열(T 차원)은 분할하지 않음
          → IPO 기업 간 독립성 가정 하에 N 분할

        Returns:
            최적 lambda_val
        """
        N, p = Z.shape

        # λ 그리드 결정
        lambda_max = self._compute_lambda_max(R_tilde, Z)
        lambda_min = lambda_max * self.lambda_min_ratio
        lambda_grid = np.geomspace(lambda_max, lambda_min, self.lambda_grid_size)

        logger.debug(f"λ 그리드: [{lambda_min:.6f}, {lambda_max:.6f}] ({self.lambda_grid_size}개)")

        # Fold 인덱스 생성
        fold_idx = np.array_split(self.rng.permutation(N), n_folds)

        cv_mse = np.zeros(len(lambda_grid))

        for li, lam in enumerate(lambda_grid):
            fold_mse = []
            for fold in range(n_folds):
                # Train / Test 분할
                test_idx  = fold_idx[fold]
                train_idx = np.concatenate([
                    fold_idx[f] for f in range(n_folds) if f != fold
                ])

                R_train = R_tilde[train_idx, :]
                Z_train = Z[train_idx, :]
                R_test  = R_tilde[test_idx, :]
                Z_test  = Z[test_idx, :]

                # Train으로 추정
                try:
                    Gamma, F, _, _, _ = self._fit_single(R_train, Z_train, lam)
                except Exception:
                    fold_mse.append(np.inf)
                    continue

                # Test OOS MSE
                pred = Z_test @ Gamma @ F.T
                mse  = np.mean((R_test - pred) ** 2)
                fold_mse.append(mse)

            cv_mse[li] = np.mean(fold_mse)
            logger.debug(f"λ={lam:.6f}: CV MSE={cv_mse[li]:.6f}")

        best_idx   = np.argmin(cv_mse)
        lambda_opt = lambda_grid[best_idx]
        return lambda_opt

    def _compute_lambda_max(
        self,
        R_tilde: np.ndarray,
        Z:       np.ndarray,
    ) -> float:
        """
        λ_max 계산: 이 값 이상이면 모든 Γⱼ = 0.

        λ_max = 2 × max_j ‖OLS_해_j‖₂

        Γ = 0일 때 Group Gradient의 최대 norm
        → 이 값보다 λ가 크면 아무것도 선택 안 됨
        """
        N, p = Z.shape
        T, k = R_tilde.shape, self.k

        # 임시 F (단위 행렬 사용)
        F_temp = np.eye(T)[:, :self.k]

        norms = []
        for j in range(p):
            Zj = Z[:, j:j+1]
            ZjtZj = float(Zj.T @ Zj) + 1e-10
            ZjtR  = Zj.T @ R_tilde   # (1 × T)
            # 단순화: F = I 가정
            u_j = (ZjtR / ZjtZj).flatten()[:self.k]
            norms.append(np.linalg.norm(u_j))

        return 2.0 * max(norms) if norms else 1.0

    # ─────────────────────────────────────────────────────
    # 초기화
    # ─────────────────────────────────────────────────────

    def _init_from_pca(
        self,
        R_tilde: np.ndarray,
        Z:       np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        PCA 기반 초기화.

        R̃의 상위 k 고유벡터 → F 초기값
        Z에 OLS 회귀 → Γ 초기값

        이유:
          랜덤 초기화 대비 수렴 빠름
          첫 번째 init에 사용
        """
        N, T = R_tilde.shape
        _, p = Z.shape
        k    = self.k

        # R̃의 SVD → 상위 k 성분
        try:
            U, S, Vt = np.linalg.svd(R_tilde, full_matrices=False)
            F = Vt[:k, :].T   # (T × k)
        except np.linalg.LinAlgError:
            F = self.rng.standard_normal((T, k))

        # Γ 초기화: Z에 대한 OLS (페널티 없음)
        # ZΓF' ≈ R̃  → Γ = Z†·R̃·F·(F'F)⁻¹
        ZtZ = Z.T @ Z + 1e-8 * np.eye(p)
        ZtR = Z.T @ R_tilde
        FtF = F.T @ F + 1e-8 * np.eye(k)
        Gamma = np.linalg.solve(ZtZ, ZtR @ F @ np.linalg.inv(FtF))  # (p × k)

        return Gamma, F

    # ─────────────────────────────────────────────────────
    # 결과 조립
    # ─────────────────────────────────────────────────────

    def _build_result(
        self,
        Gamma:      np.ndarray,
        F:          np.ndarray,
        Z:          np.ndarray,
        R_tilde:    np.ndarray,
        n_iter:     int,
        converged:  bool,
        obj_hist:   list[float],
        lambda_val: float,
    ) -> SparsePPCAResult:
        """추정 결과를 SparsePPCAResult로 조립"""

        # 선택된 특성 (Γⱼ ≠ 0인 행)
        row_norms = np.array([
            np.linalg.norm(Gamma[j, :])
            for j in range(Gamma.shape[0])
        ])
        selected = list(np.where(row_norms > 1e-10)[0])

        # 팩터 로딩 Λ = ZΓ
        Lambda = Z @ Gamma

        # R² 계산
        ss_res = np.sum((R_tilde - Lambda @ F.T) ** 2)
        ss_tot = np.sum((R_tilde - R_tilde.mean()) ** 2)
        r2     = 1.0 - ss_res / (ss_tot + 1e-10)

        return SparsePPCAResult(
            Gamma=Gamma,
            F=F,
            Lambda=Lambda,
            n_iter=n_iter,
            converged=converged,
            obj_history=obj_hist,
            lambda_opt=lambda_val,
            selected_features=selected,
            n_selected=len(selected),
            r2_total=max(0.0, r2),
        )

    # ─────────────────────────────────────────────────────
    # 유틸리티
    # ─────────────────────────────────────────────────────

    def _standardize(self, Z: np.ndarray) -> np.ndarray:
        """
        특성 행렬 표준화 (μ=0, σ=1).

        이유:
          Group LASSO 페널티가 스케일에 민감
          → 모든 특성을 동등한 기준으로 비교하기 위해
        """
        mu  = Z.mean(axis=0)
        std = Z.std(axis=0)
        std[std < 1e-10] = 1.0  # 상수 특성 보호
        return (Z - mu) / std

    def select_k(
        self,
        R:      np.ndarray,
        Z:      np.ndarray,
        k_max:  int = 8,
        criterion: str = "IC1",
    ) -> int:
        """
        팩터 수 k를 Information Criterion으로 결정.

        Bai & Ng(2002) IC1:
          IC1(k) = ln(RSS(k)/NT) + k·(N+T)/(NT)·ln(NT/(N+T))

        사용법:
          k_opt = model.select_k(R, Z, k_max=8)
          model.k = k_opt
          result = model.fit(R, Z)
        """
        N, T = R.shape
        _, p = Z.shape
        _, R_tilde = self._project(R, self._standardize(Z))

        ic_values = []
        for k in range(1, k_max + 1):
            self.k = k
            Gamma, F, _, _, _ = self._fit_single(R_tilde, self._standardize(Z), lambda_val=0.0)
            rss = np.sum((R_tilde - Z @ Gamma @ F.T) ** 2)
            ic  = np.log(rss / (N * T)) + k * (N + T) / (N * T) * np.log(N * T / (N + T))
            ic_values.append(ic)
            logger.debug(f"k={k}: IC1={ic:.4f}")

        k_opt = int(np.argmin(ic_values)) + 1
        self.k = k_opt  # 원복
        logger.info(f"최적 팩터 수 k={k_opt} (IC1 기준)")
        return k_opt
