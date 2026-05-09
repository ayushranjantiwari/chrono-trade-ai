import warnings
warnings.filterwarnings('ignore')

import time

from src.pipeline.pipeline import run_pipeline
from src.pipeline.predict_future import predict_next_5_days
from src.backtests.weekly_evaluation import evaluate_last_week


def main():

    start_time = time.time()

    print("\n" + "="*60)
    print("🚀 TIME SERIES FORECASTING SYSTEM")
    print("="*60)

    # =====================
    # PHASE 1: TRAINING
    # =====================
    try:
        print("\n=== Phase 1: Model Training & Evaluation ===")

        t0 = time.time()
        results_df = run_pipeline()
        t1 = time.time()

        print(f"\n✅ Phase 1 Completed ({round(t1 - t0, 2)} sec)")

    except Exception as e:
        print("\n❌ Error in Phase 1:", str(e))
        return

    # =====================
    # PHASE 2: PREDICTION
    # =====================
    try:
        print("\n=== Phase 2: Future Prediction (Next 5 Days) ===")

        t0 = time.time()
        future_df = predict_next_5_days()
        t1 = time.time()

        print(f"\n✅ Predictions Generated ({round(t1 - t0, 2)} sec)")

    except Exception as e:
        print("\n❌ Error in Phase 2:", str(e))
        return

    # =====================
    # PHASE 3: EVALUATION
    # =====================
    try:
        print("\n=== Phase 3: Weekly Evaluation ===")

        t0 = time.time()
        eval_df = evaluate_last_week()
        t1 = time.time()

        if eval_df is not None:
            print(f"\n✅ Evaluation Completed ({round(t1 - t0, 2)} sec)")
        else:
            print("\n⚠ Evaluation skipped (no overlapping data yet)")

    except Exception as e:
        print("\n❌ Error in Phase 3:", str(e))
        return

    # =====================
    # FINAL SUMMARY
    # =====================
    total_time = time.time() - start_time

    print("\n" + "="*60)
    print("🎯 PIPELINE EXECUTION COMPLETE")
    print(f"⏱ Total Time: {round(total_time, 2)} seconds")
    print("="*60)


# =====================
# ENTRY POINT
# =====================
if __name__ == "__main__":
    main()