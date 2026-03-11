"""
=============================================================
 Malaria Detection – Real-World Testing Scenarios
=============================================================
Scenario 1: Rural Clinic Screening (batch, low-resource)
Scenario 2: Hospital Mass Screening  (high-throughput)
Scenario 3: Research Analysis        (detailed metrics)
=============================================================
"""

import os
import sys
import time
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))
from prediction.predictor import MalariaPredictor, PredictionResult


# ─── Shared helpers ───────────────────────────────────────
def _load_test_images(folder: str, n: int = None):
    folder = Path(folder)
    ext    = {".png", ".jpg", ".jpeg", ".tif"}
    images = [f for f in folder.rglob("*") if f.suffix.lower() in ext]
    if n:
        images = random.sample(images, min(n, len(images)))
    return images


def _print_header(title: str, width: int = 60):
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def _risk_color(label: str) -> str:
    return "\033[91m🔴\033[0m" if label == "Parasitized" else "\033[92m🟢\033[0m"


# ═══════════════════════════════════════════════════════════
#  SCENARIO 1 – Rural Clinic Screening
# ═══════════════════════════════════════════════════════════
def scenario_rural_clinic(predictor: MalariaPredictor,
                           test_folder: str, n_patients: int = 20):
    """
    Simulates a low-resource rural clinic processing
    one blood smear per patient with basic triage output.
    """
    _print_header("SCENARIO 1 — Rural Clinic Screening", 60)
    print(f"  Date       : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Patients   : {n_patients}")
    print(f"  Threshold  : {predictor.threshold}")

    images = _load_test_images(test_folder, n_patients)
    if not images:
        print("  ⚠️  No test images found.")
        return

    results   = []
    flagged   = []

    print(f"\n  {'ID':<6} {'File':<30} {'Result':<16} {'Conf%':>6}  {'ms':>6}")
    print("  " + "─" * 66)

    for pid, img_path in enumerate(images, 1):
        r = predictor.predict_single(img_path, img_path.name)
        results.append(r)
        if r.is_parasitized:
            flagged.append((pid, r))

        status = _risk_color(r.label)
        print(f"  P{pid:03d}   {r.filename:<30} "
              f"{status} {r.label:<12} {r.confidence*100:6.1f}%  {r.inference_ms:6.1f}")

    # Triage summary
    n_pos  = len(flagged)
    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │  TRIAGE SUMMARY                     │")
    print(f"  │  Screened     : {n_patients:<5} patients        │")
    print(f"  │  🔴 Flagged   : {n_pos:<5} ({n_pos/n_patients*100:.0f}%)             │")
    print(f"  │  🟢 Clear     : {n_patients-n_pos:<5} ({(n_patients-n_pos)/n_patients*100:.0f}%)             │")
    print(f"  │  Avg speed    : {np.mean([r.inference_ms for r in results]):.0f} ms/image           │")
    print(f"  └─────────────────────────────────────┘")

    if flagged:
        print(f"\n  ⚠️  Refer to microscopy: patients "
              + ", ".join(f"P{pid:03d}" for pid, _ in flagged))

    # Save JSON report
    report = {
        "scenario":    "rural_clinic",
        "date":        datetime.now().isoformat(),
        "total":       n_patients,
        "positive":    n_pos,
        "negative":    n_patients - n_pos,
        "prevalence":  round(n_pos / n_patients * 100, 2),
        "results":     [r.to_dict() for r in results],
    }
    os.makedirs("reports", exist_ok=True)
    with open("reports/rural_clinic_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  📄  Report → reports/rural_clinic_report.json")
    return results


# ═══════════════════════════════════════════════════════════
#  SCENARIO 2 – Hospital Mass Screening
# ═══════════════════════════════════════════════════════════
def scenario_hospital_mass(predictor: MalariaPredictor,
                            test_folder: str, n_samples: int = 200):
    """
    High-throughput hospital workflow: batch processing with
    priority queue and performance benchmarking.
    """
    _print_header("SCENARIO 2 — Hospital Mass Screening", 60)
    print(f"  Target samples  : {n_samples}")
    print(f"  Processing mode : batch (all images)")

    images = _load_test_images(test_folder, n_samples)
    print(f"  Found images    : {len(images)}\n")

    # Batch predict
    t0      = time.perf_counter()
    results = predictor.predict_batch(images, [i.name for i in images])
    elapsed = time.perf_counter() - t0

    positives = [r for r in results if r.is_parasitized]
    negatives = [r for r in results if not r.is_parasitized]

    # Confidence bucketing
    buckets = {"HIGH (>90%)": 0, "MED (70–90%)": 0,
               "LOW (50–70%)": 0, "AMBIGUOUS (<70%)": 0}
    for r in results:
        c = r.confidence
        if c > 0.90:    buckets["HIGH (>90%)"]    += 1
        elif c > 0.70:  buckets["MED (70–90%)"]   += 1
        elif c > 0.50:  buckets["LOW (50–70%)"]   += 1
        else:           buckets["AMBIGUOUS (<70%)"] += 1

    throughput = len(results) / elapsed

    print(f"  ┌──────────────────────────────────────────┐")
    print(f"  │  MASS SCREENING RESULTS                  │")
    print(f"  │                                          │")
    print(f"  │  Total processed  : {len(results):<6}                 │")
    print(f"  │  🔴 Parasitized   : {len(positives):<6} ({len(positives)/len(results)*100:.1f}%)         │")
    print(f"  │  🟢 Uninfected    : {len(negatives):<6} ({len(negatives)/len(results)*100:.1f}%)         │")
    print(f"  │                                          │")
    print(f"  │  Throughput       : {throughput:.1f} img/sec           │")
    print(f"  │  Total time       : {elapsed:.2f} sec                  │")
    print(f"  │  Avg latency      : {elapsed/len(results)*1000:.1f} ms/img            │")
    print(f"  └──────────────────────────────────────────┘")

    print(f"\n  Confidence distribution:")
    for bucket, count in buckets.items():
        bar = "█" * int(count / len(results) * 40)
        print(f"    {bucket:<20} {bar} {count:>4}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Hospital Mass Screening – Analytics", fontsize=13, fontweight="bold")

    # Pie
    axes[0].pie([len(positives), len(negatives)],
                labels=["Parasitized", "Uninfected"],
                colors=["#F44336", "#4CAF50"],
                autopct="%1.1f%%", startangle=90,
                wedgeprops=dict(edgecolor="white", linewidth=2))
    axes[0].set_title("Prevalence")

    # Confidence histogram
    confs = [r.confidence for r in results]
    axes[1].hist(confs, bins=20, color="#2196F3", edgecolor="white")
    axes[1].axvline(np.mean(confs), color="red", linestyle="--",
                    label=f"Mean={np.mean(confs):.2f}")
    axes[1].set_title("Confidence Distribution")
    axes[1].set_xlabel("Confidence"); axes[1].legend()

    # Inference time
    ms_vals = [r.inference_ms for r in results]
    axes[2].hist(ms_vals, bins=20, color="#FF9800", edgecolor="white")
    axes[2].axvline(np.mean(ms_vals), color="red", linestyle="--",
                    label=f"Mean={np.mean(ms_vals):.1f}ms")
    axes[2].set_title("Inference Latency (ms)")
    axes[2].set_xlabel("ms"); axes[2].legend()

    plt.tight_layout()
    os.makedirs("reports", exist_ok=True)
    plt.savefig("reports/hospital_mass_screening.png", dpi=150)
    plt.show()
    print("  📊  Plot → reports/hospital_mass_screening.png")
    return results


# ═══════════════════════════════════════════════════════════
#  SCENARIO 3 – Research Analysis
# ═══════════════════════════════════════════════════════════
def scenario_research_analysis(predictor: MalariaPredictor,
                                test_folder: str,
                                ground_truth: Dict[str, int] = None):
    """
    Detailed research-grade analysis with per-class breakdown,
    error analysis, and threshold sensitivity study.
    """
    _print_header("SCENARIO 3 — Research Analysis", 60)

    images  = _load_test_images(test_folder)
    results = predictor.predict_batch(images, [i.name for i in images])

    probas   = np.array([r.raw_proba for r in results])
    confs    = np.array([r.confidence for r in results])

    if ground_truth:
        y_true  = np.array([ground_truth.get(r.filename, -1) for r in results])
        valid   = y_true >= 0
        y_true  = y_true[valid]
        y_proba = probas[valid]

        # Threshold sensitivity
        thresholds  = np.linspace(0.1, 0.9, 81)
        sensitivities, specificities, f1s = [], [], []

        from sklearn.metrics import f1_score as f1_fn, confusion_matrix as cm_fn
        for t in thresholds:
            y_p = (y_proba >= t).astype(int)
            tn, fp, fn, tp = cm_fn(y_true, y_p).ravel()
            sensitivities.append(tp / (tp + fn + 1e-9))
            specificities.append(tn / (tn + fp + 1e-9))
            f1s.append(f1_fn(y_true, y_p, zero_division=0))

        best_t = thresholds[np.argmax(f1s)]
        print(f"\n  Optimal threshold (max F1): {best_t:.2f}  (F1={max(f1s):.4f})")

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(thresholds, sensitivities, label="Sensitivity", color="#F44336")
        axes[0].plot(thresholds, specificities, label="Specificity", color="#2196F3")
        axes[0].plot(thresholds, f1s,           label="F1-Score",    color="#4CAF50")
        axes[0].axvline(best_t, color="black", linestyle="--", label=f"Best t={best_t:.2f}")
        axes[0].set(title="Threshold Sensitivity Analysis",
                    xlabel="Threshold", ylabel="Score")
        axes[0].legend(); axes[0].grid(alpha=0.3)

        # Probability calibration
        bins = np.linspace(0, 1, 11)
        bin_means, bin_fracs = [], []
        for i in range(len(bins) - 1):
            mask = (y_proba >= bins[i]) & (y_proba < bins[i+1])
            if mask.sum() > 0:
                bin_means.append(y_proba[mask].mean())
                bin_fracs.append(y_true[mask].mean())

        axes[1].plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        axes[1].plot(bin_means, bin_fracs, "o-", color="#9C27B0",
                     label="Model calibration")
        axes[1].set(title="Probability Calibration",
                    xlabel="Mean predicted probability",
                    ylabel="Fraction of positives")
        axes[1].legend(); axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("reports/research_analysis.png", dpi=150)
        plt.show()
        print("  📊  Plot → reports/research_analysis.png")

    else:
        # Blind analysis (no GT)
        print(f"\n  Total images     : {len(results)}")
        print(f"  Parasitized pred : {sum(r.is_parasitized for r in results)}")
        print(f"  Uninfected pred  : {sum(not r.is_parasitized for r in results)}")
        print(f"  Mean confidence  : {confs.mean():.4f}")
        print(f"  Std confidence   : {confs.std():.4f}")
        print(f"  Min confidence   : {confs.min():.4f}")
        print(f"  Max confidence   : {confs.max():.4f}")
        print(f"\n  Low-confidence predictions (<70%): "
              f"{(confs < 0.7).sum()} images → recommend manual review")


# ─── Main runner ──────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   default="saved_models/best_model.keras")
    ap.add_argument("--data",    default="data/val")
    ap.add_argument("--scenario", default="all",
                    choices=["all", "rural", "hospital", "research"])
    args = ap.parse_args()

    pred = MalariaPredictor(args.model)

    if args.scenario in ("all", "rural"):
        scenario_rural_clinic(pred, args.data, n_patients=20)

    if args.scenario in ("all", "hospital"):
        scenario_hospital_mass(pred, args.data, n_samples=100)

    if args.scenario in ("all", "research"):
        scenario_research_analysis(pred, args.data)
