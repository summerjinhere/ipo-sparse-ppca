"""
run_pipeline.py
────────────────
IPO Overhang Sparse PPCA — 전체 파이프라인 실행기

실행 방법:
  python run_pipeline.py --stage all
  python run_pipeline.py --stage ingest
  python run_pipeline.py --stage extract
  python run_pipeline.py --stage model
  python run_pipeline.py --stage evaluate

파이프라인 단계:
  Stage 1 ingest:   EDGAR 크롤링 + HTML 저장
  Stage 2 extract:  수치 추출 + LLM 어노테이션
  Stage 3 features: 특성 행렬 Z 구성
  Stage 4 returns:  이벤트 타임 수익률 행렬 R 구성
  Stage 5 model:    Sparse PPCA 추정
  Stage 6 evaluate: Span Test + CAR 분석
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

from src.utils.config import Config, setup_logger, ROOT
from src.ingestion.edgar_crawler import EdgarCrawler
from src.features.numerical_extractor import NumericalExtractor
from src.models.sparse_ppca import SparsePPCA
from src.evaluation.span_test import SpanTest


def run_ingest(cfg: Config) -> None:
    """
    Stage 1: EDGAR 크롤링.

    출력:
      data/interim/filing_manifest.json
      data/raw/html/424B4/*.html
      data/raw/html/S-1/*.html
    """
    logger.info("=" * 50)
    logger.info("Stage 1: EDGAR 크롤링 시작")
    logger.info("=" * 50)

    crawler = EdgarCrawler(cfg)

    # Filing 목록 수집
    filings = list(crawler.iter_filings(
        form_types=["424B4", "S-1"],
        start=f"{cfg.data.sample.start_year}-01-01",
        end=f"{cfg.data.sample.end_year}-12-31",
    ))
    logger.info(f"Filing 목록 수집: {len(filings)}건")

    # HTML 다운로드
    downloaded = crawler.download_all(filings)

    # Manifest 저장
    manifest_path = crawler.save_manifest(downloaded)
    logger.info(f"Stage 1 완료: {manifest_path}")


def run_extract(cfg: Config) -> None:
    """
    Stage 2: 수치 추출.

    입력:  data/interim/filing_manifest.json
    출력:  data/interim/numerical_fields.csv
    """
    logger.info("=" * 50)
    logger.info("Stage 2: 수치 추출 시작")
    logger.info("=" * 50)

    manifest_path = ROOT / cfg.paths.interim / "filing_manifest.json"
    if not manifest_path.exists():
        logger.error(f"Manifest 없음: {manifest_path}. 먼저 Stage 1 실행.")
        return

    filings = EdgarCrawler.load_manifest(manifest_path)
    # 424B4만 추출
    b4_filings = [f for f in filings if f.form_type == "424B4" and f.downloaded]
    logger.info(f"424B4 Filing: {len(b4_filings)}건")

    results = []
    failed  = []

    for filing in b4_filings:
        html_path = ROOT / cfg.paths.raw_html / "424B4" / f"{filing.cik}_{filing.accession_no[:8]}.html"
        if not html_path.exists():
            failed.append(filing.cik)
            continue

        try:
            html = html_path.read_text(encoding="utf-8")
            extractor = NumericalExtractor(
                html=html,
                meta={
                    "cik":          filing.cik,
                    "company_name": filing.company_name,
                    "filed_date":   filing.filed_date,
                    "accession_no": filing.accession_no,
                }
            )
            fields = extractor.extract()
            fields.compute_derived()
            qc_ok  = fields.qc_check()

            if not qc_ok:
                logger.warning(
                    f"QC 실패: {filing.company_name} "
                    f"(null: {fields.null_fields})"
                )
            results.append(fields)

        except Exception as e:
            logger.error(f"추출 오류 ({filing.company_name}): {e}")
            failed.append(filing.cik)

    # CSV 저장
    out_path = ROOT / cfg.paths.interim / "numerical_fields.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # dataclass → dict
    records = []
    for f in results:
        d = {k: v for k, v in f.__dict__.items()
             if not isinstance(v, list)}
        records.append(d)

    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)

    logger.info(
        f"Stage 2 완료: {len(results)}건 성공 / "
        f"{len(failed)}건 실패 → {out_path}"
    )
    logger.info(f"핵심 필드 null rate:\n{df[['offer_price','lock_up_days','unlock_pressure']].isnull().mean()}")


def run_model(cfg: Config) -> None:
    """
    Stage 5: Sparse PPCA 추정.

    입력:  data/processed/Z_matrix.npy  (N × p)
           data/processed/R_matrix.npy  (N × T)
           data/processed/feature_names.json
    출력:  data/results/sparse_ppca_result.npz
    """
    logger.info("=" * 50)
    logger.info("Stage 5: Sparse PPCA 추정 시작")
    logger.info("=" * 50)

    proc_dir = ROOT / cfg.paths.processed
    Z_path = proc_dir / "Z_matrix.npy"
    R_path = proc_dir / "R_matrix.npy"

    if not Z_path.exists() or not R_path.exists():
        logger.error("특성/수익률 행렬 없음. Stage 3-4 먼저 실행.")
        return

    Z = np.load(Z_path)
    R = np.load(R_path)
    with open(proc_dir / "feature_names.json") as f:
        feature_names = json.load(f)

    logger.info(f"R shape: {R.shape}, Z shape: {Z.shape}")

    # 팩터 수 결정
    model = SparsePPCA(
        k=cfg.model.ppca.k_max,
        max_iter=cfg.model.ppca.max_iter,
        tol=cfg.model.ppca.tol,
        n_init=cfg.model.ppca.n_init,
        lambda_grid_size=cfg.model.sparse.lambda_grid_size,
        seed=cfg.project.seed,
    )
    k_opt = model.select_k(R, Z, k_max=cfg.model.ppca.k_max)
    model.k = k_opt

    # 추정
    result = model.fit(R, Z)

    # 결과 저장
    res_dir = ROOT / cfg.paths.results
    res_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        res_dir / "sparse_ppca_result.npz",
        Gamma=result.Gamma,
        F=result.F,
        Lambda=result.Lambda,
        selected_features=np.array(result.selected_features),
    )

    # 선택된 특성 이름 출력
    selected_names = [feature_names[i] for i in result.selected_features
                      if i < len(feature_names)]
    logger.info(f"선택된 특성: {selected_names}")

    # 요약 저장
    summary = {
        "k_opt":            k_opt,
        "lambda_opt":       result.lambda_opt,
        "n_selected":       result.n_selected,
        "selected_features": result.selected_features,
        "selected_names":   selected_names,
        "r2_total":         result.r2_total,
        "converged":        result.converged,
        "n_iter":           result.n_iter,
    }
    with open(res_dir / "model_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Stage 5 완료 → {res_dir}")


def run_evaluate(cfg: Config) -> None:
    """
    Stage 6: Span Test + CAR 분석.

    입력:  data/results/sparse_ppca_result.npz
           data/processed/R_matrix.npy
           data/processed/market_returns.npy (선택)
    출력:  data/results/span_test_result.json
    """
    logger.info("=" * 50)
    logger.info("Stage 6: Span Test 시작")
    logger.info("=" * 50)

    res_dir  = ROOT / cfg.paths.results
    proc_dir = ROOT / cfg.paths.processed

    # 로드
    ppca_res = np.load(res_dir / "sparse_ppca_result.npz")
    F_hat    = ppca_res["F"]   # (T × k)
    R        = np.load(proc_dir / "R_matrix.npy")

    # 시장 수익률 (있으면)
    mkt_path = proc_dir / "market_returns.npy"
    market_ret = np.load(mkt_path) if mkt_path.exists() else None

    # CAR 계산
    tester = SpanTest(significance_level=cfg.evaluation.span_test.significance_level)

    # 이벤트 타임: t=0 = 락업 해제일
    event_col = cfg.data.event_window.pre_days   # 예: 90

    all_results = {}
    for window in cfg.evaluation.car_windows:
        win = tuple(window)
        car = tester.compute_car(
            returns=R,
            event_col=event_col,
            window=win,
            market_ret=market_ret,
        )

        # 팩터 스코어 (N × k): 기업별 팩터 로딩
        # F_hat = (T × k) → N 방향 스코어는 Lambda = ZΓ
        Lambda = ppca_res["Lambda"]   # (N × k)

        # Span Test
        result = tester.test(
            car=car,
            F_scores=Lambda,
            boot_reps=cfg.evaluation.span_test.bootstrap_reps,
        )

        all_results[str(win)] = {
            "F_stat":        result.F_stat,
            "p_value":       result.p_value,
            "boot_p_value":  result.boot_p_value,
            "rejected":      result.rejected,
            "r2":            result.r2,
            "alpha":         result.alpha,
            "alpha_t":       result.alpha_t,
            "gamma":         result.gamma.tolist(),
            "t_stats":       result.t_stats.tolist(),
        }

    # 저장
    out_path = res_dir / "span_test_result.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Stage 6 완료 → {out_path}")


# ── 메인 ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="IPO Overhang Sparse PPCA Pipeline")
    parser.add_argument(
        "--stage",
        choices=["all", "ingest", "extract", "model", "evaluate"],
        default="all",
        help="실행할 파이프라인 단계",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="config YAML 경로 (기본: configs/config.yaml)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    # 설정 로드
    cfg = Config.load(args.config)
    setup_logger(
        log_file=f"logs/pipeline_{args.stage}.log",
        level=args.log_level,
    )

    logger.info(f"IPO Overhang Sparse PPCA — Stage: {args.stage}")
    logger.info(f"샘플 기간: {cfg.data.sample.start_year}–{cfg.data.sample.end_year}")

    # 단계별 실행
    stage_map = {
        "ingest":   run_ingest,
        "extract":  run_extract,
        "model":    run_model,
        "evaluate": run_evaluate,
    }

    if args.stage == "all":
        for name, fn in stage_map.items():
            fn(cfg)
    else:
        stage_map[args.stage](cfg)


if __name__ == "__main__":
    main()
