import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from inference.predict_unified import (
        diagnose_error_type,
        evaluate_logic_checker,
        evaluate_mouth_open_during_hold,
        infer_one,
    )
    from models.registry import get_device, load_model
    from preprocessing.extract_features_enhanced import extract as extract_features_enhanced
    from utils.paths import project_paths
except ModuleNotFoundError:
    src_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(src_root))
    from inference.predict_unified import (
        diagnose_error_type,
        evaluate_logic_checker,
        evaluate_mouth_open_during_hold,
        infer_one,
    )
    from models.registry import get_device, load_model
    from preprocessing.extract_features_enhanced import extract as extract_features_enhanced
    from utils.paths import project_paths


MODEL_ROW_COLUMNS = [
    "model",
    "video_rel",
    "feature_path",
    "compact_sequence",
    "core_order_ok",
    "core_breath_hold_ok",
    "core_verdict",
    "mouth_open_during_hold",
    "final_is_correct",
    "final_verdict",
    "error_type",
    "error_step",
    "diagnosis_confidence",
    "reason",
    "avg_zadrzeni_sec",
    "zadrzeni_count",
    "mouth_avg_hold",
    "mouth_avg_all",
    "mouth_delta",
    "mouth_ratio",
    "pred_mean_max_prob",
    "pred_p10_max_prob",
    "pred_min_max_prob",
]


CONSENSUS_COLUMNS = [
    "video_rel",
    "asformer_error_type",
    "asformer_error_step",
    "asformer_final_verdict",
    "mstcn_error_type",
    "mstcn_error_step",
    "mstcn_final_verdict",
    "agree_error_type",
    "agree_error_step",
    "agree_final_verdict",
]


SUMMARY_COLUMNS = [
    "model",
    "videos",
    "predicted_correct_count",
    "predicted_wrong_count",
    "predicted_correct_pct",
    "predicted_wrong_pct",
]


def resolve_path(path_str, root):
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p.resolve()

    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    root_candidate = (root / p).resolve()
    if root_candidate.exists():
        return root_candidate

    return (root / p).resolve()


def find_feature_files(features_dir):
    files = []
    for root, _, names in os.walk(features_dir):
        for name in names:
            if name.endswith(".npy"):
                files.append(os.path.join(root, name))
    files.sort()
    return files


def write_csv(path, rows, columns):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def build_model_row(model_name, feat_path, feature_rel, prediction, pred_conf, logic, mouth, diag):
    final_correct = int(diag.get("is_correct", 0))
    return {
        "model": model_name,
        "video_rel": feature_rel.replace(".npy", ""),
        "feature_path": feat_path,
        "compact_sequence": str(logic.get("compact_sequence", [])),
        "core_order_ok": int(bool(logic.get("order_ok", False))),
        "core_breath_hold_ok": int(bool(logic.get("breath_hold_ok", False))),
        "core_verdict": "PROSEL" if bool(logic.get("overall_ok", False)) else "NEPROSEL",
        "mouth_open_during_hold": int(bool(mouth.get("is_open_mouth_hold", False))),
        "final_is_correct": final_correct,
        "final_verdict": "PROSEL" if final_correct == 1 else "NEPROSEL",
        "error_type": diag.get("error_type", ""),
        "error_step": diag.get("error_step", ""),
        "diagnosis_confidence": float(diag.get("confidence", 0.0)),
        "reason": diag.get("reason", ""),
        "avg_zadrzeni_sec": float(logic.get("avg_zadrzeni_sec", 0.0)),
        "zadrzeni_count": int(logic.get("zadrzeni_count", 0)),
        "mouth_avg_hold": float(mouth.get("avg_hold_mouth", 0.0)),
        "mouth_avg_all": float(mouth.get("avg_all_mouth", 0.0)),
        "mouth_delta": float(mouth.get("delta", 0.0)),
        "mouth_ratio": float(mouth.get("relative_ratio", 0.0)),
        "pred_mean_max_prob": float(pred_conf.get("mean_max_prob", 0.0)),
        "pred_p10_max_prob": float(pred_conf.get("p10_max_prob", 0.0)),
        "pred_min_max_prob": float(pred_conf.get("min_max_prob", 0.0)),
    }


def aggregate_consensus(model_rows):
    by_video = defaultdict(dict)
    for row in model_rows:
        by_video[row["video_rel"]][row["model"]] = row

    out = []
    for video_rel in sorted(by_video.keys()):
        pair = by_video[video_rel]
        asf = pair.get("asformer")
        mst = pair.get("mstcn")
        if asf is None or mst is None:
            continue

        out.append(
            {
                "video_rel": video_rel,
                "asformer_error_type": asf["error_type"],
                "asformer_error_step": asf["error_step"],
                "asformer_final_verdict": asf["final_verdict"],
                "mstcn_error_type": mst["error_type"],
                "mstcn_error_step": mst["error_step"],
                "mstcn_final_verdict": mst["final_verdict"],
                "agree_error_type": int(asf["error_type"] == mst["error_type"]),
                "agree_error_step": int(asf["error_step"] == mst["error_step"]),
                "agree_final_verdict": int(asf["final_verdict"] == mst["final_verdict"]),
            }
        )
    return out


def aggregate_summary(model_rows):
    out = []
    for model in ["asformer", "mstcn"]:
        rows = [r for r in model_rows if r["model"] == model]
        n = len(rows)
        if n == 0:
            continue
        correct = sum(int(r["final_is_correct"]) for r in rows)
        wrong = n - correct
        out.append(
            {
                "model": model,
                "videos": n,
                "predicted_correct_count": correct,
                "predicted_wrong_count": wrong,
                "predicted_correct_pct": (100.0 * correct / n),
                "predicted_wrong_pct": (100.0 * wrong / n),
            }
        )
    return out


def plot_predicted_correctness(summary_rows, out_png):
    if not summary_rows:
        return

    models = [r["model"] for r in summary_rows]
    correct = [r["predicted_correct_count"] for r in summary_rows]
    wrong = [r["predicted_wrong_count"] for r in summary_rows]
    x = np.arange(len(models))

    plt.figure(figsize=(8, 5))
    plt.bar(x, correct, label="Predicted correct")
    plt.bar(x, wrong, bottom=correct, label="Predicted wrong")
    plt.xticks(x, models)
    plt.ylabel("Videos")
    plt.title("Unseen Videos: Predicted Correct vs Wrong")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close()


def plot_error_type_distribution(model_rows, out_png):
    if not model_rows:
        return

    per_model_counter = {
        "asformer": Counter(),
        "mstcn": Counter(),
    }
    all_error_types = set()

    for row in model_rows:
        model = row["model"]
        e = row["error_type"]
        per_model_counter[model][e] += 1
        all_error_types.add(e)

    error_types = sorted(all_error_types)
    x = np.arange(len(error_types))
    width = 0.36
    asf_vals = [per_model_counter["asformer"].get(e, 0) for e in error_types]
    mst_vals = [per_model_counter["mstcn"].get(e, 0) for e in error_types]

    plt.figure(figsize=(max(10, len(error_types) * 1.2), 5))
    plt.bar(x - width / 2, asf_vals, width=width, label="ASFormer")
    plt.bar(x + width / 2, mst_vals, width=width, label="MS-TCN")
    plt.xticks(x, error_types, rotation=30, ha="right")
    plt.ylabel("Videos")
    plt.title("Unseen Videos: Predicted Error Type Distribution")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close()


def plot_consensus(consensus_rows, out_png):
    if not consensus_rows:
        return

    labels = ["Error type agreement", "Error step agreement", "Final verdict agreement"]
    values = [
        100.0 * np.mean([r["agree_error_type"] for r in consensus_rows]),
        100.0 * np.mean([r["agree_error_step"] for r in consensus_rows]),
        100.0 * np.mean([r["agree_final_verdict"] for r in consensus_rows]),
    ]

    x = np.arange(len(labels))
    plt.figure(figsize=(8, 5))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=12, ha="right")
    plt.ylim(0, 100)
    plt.ylabel("Agreement (%)")
    plt.title("ASFormer vs MS-TCN Agreement on Unseen Videos")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close()


def main():
    paths = project_paths(__file__)
    root = paths["root"]
    default_raw = root / "data" / "raw_videos_unseen"
    default_out_base = root / "results" / "thesis_report" / "unseen_report"
    default_features = default_out_base / "features_extracted"

    parser = argparse.ArgumentParser(
        description="Batch pipeline for unseen raw videos: extract features, infer both models, export thesis-ready tables and plots."
    )
    parser.add_argument("--raw_dir", default=str(default_raw), help="Slozka s nevidenymi raw videi")
    parser.add_argument(
        "--features_dir",
        default=str(default_features),
        help="Kam ulozit extrahovane .npy features pro unseen data",
    )
    parser.add_argument("--asformer_ckpt", default="src/training/asformer_attention_v1.pth")
    parser.add_argument("--mstcn_ckpt", default="src/training/mstcn_v1.pth")
    parser.add_argument("--out_dir", default=None, help="Vystupni slozka reportu")
    parser.add_argument("--overwrite_features", action="store_true")
    parser.add_argument("--min-breath-hold-sec", type=float, default=4.5)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--mouth-open-threshold", type=float, default=0.008)
    parser.add_argument("--mouth-open-ratio-threshold", type=float, default=1.25)
    parser.add_argument("--mouth-open-delta-threshold", type=float, default=0.002)
    args = parser.parse_args()

    raw_dir = resolve_path(args.raw_dir, root)
    features_dir = resolve_path(args.features_dir, root)
    asf_ckpt = str(resolve_path(args.asformer_ckpt, root))
    mst_ckpt = str(resolve_path(args.mstcn_ckpt, root))

    if args.out_dir:
        out_dir = resolve_path(args.out_dir, root)
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = (default_out_base / f"run_{run_id}").resolve()

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw unseen directory not found: {raw_dir}")

    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[1/4] Extracting features from unseen raw videos: {raw_dir}")
    extract_features_enhanced(
        input_root=str(raw_dir),
        output_root=str(features_dir),
        overwrite=bool(args.overwrite_features),
    )

    feat_files = find_feature_files(str(features_dir))
    if not feat_files:
        raise RuntimeError(f"No .npy feature files found in: {features_dir}")

    print(f"[2/4] Loading models on device")
    device = get_device()
    asf_model, asf_ckpt_path = load_model("asformer", checkpoint_path=asf_ckpt, device=device)
    mst_model, mst_ckpt_path = load_model("mstcn", checkpoint_path=mst_ckpt, device=device)

    print(f"[3/4] Running inference+diagnosis on {len(feat_files)} videos")
    model_rows = []
    for feat_path in feat_files:
        feat_rel = os.path.relpath(feat_path, str(features_dir)).replace("\\", "/")
        features_t_by_c = np.load(feat_path)

        asf_pred, asf_conf = infer_one("asformer", asf_model, feat_path, device)
        asf_logic = evaluate_logic_checker(asf_pred, fps=args.fps, min_breath_hold_sec=args.min_breath_hold_sec)
        asf_mouth = evaluate_mouth_open_during_hold(
            asf_pred,
            features_t_by_c,
            mouth_open_threshold=args.mouth_open_threshold,
            mouth_open_ratio_threshold=args.mouth_open_ratio_threshold,
            mouth_open_delta_threshold=args.mouth_open_delta_threshold,
        )
        asf_diag = diagnose_error_type(asf_logic, args.min_breath_hold_sec, mouth_report=asf_mouth)
        model_rows.append(
            build_model_row(
                "asformer",
                feat_path,
                feat_rel,
                asf_pred,
                asf_conf,
                asf_logic,
                asf_mouth,
                asf_diag,
            )
        )

        mst_pred, mst_conf = infer_one("mstcn", mst_model, feat_path, device)
        mst_logic = evaluate_logic_checker(mst_pred, fps=args.fps, min_breath_hold_sec=args.min_breath_hold_sec)
        mst_mouth = evaluate_mouth_open_during_hold(
            mst_pred,
            features_t_by_c,
            mouth_open_threshold=args.mouth_open_threshold,
            mouth_open_ratio_threshold=args.mouth_open_ratio_threshold,
            mouth_open_delta_threshold=args.mouth_open_delta_threshold,
        )
        mst_diag = diagnose_error_type(mst_logic, args.min_breath_hold_sec, mouth_report=mst_mouth)
        model_rows.append(
            build_model_row(
                "mstcn",
                feat_path,
                feat_rel,
                mst_pred,
                mst_conf,
                mst_logic,
                mst_mouth,
                mst_diag,
            )
        )

    consensus_rows = aggregate_consensus(model_rows)
    summary_rows = aggregate_summary(model_rows)

    print(f"[4/4] Writing thesis-ready tables and plots to: {out_dir}")
    per_video_csv = out_dir / "unseen_per_video_predictions.csv"
    consensus_csv = out_dir / "unseen_model_consensus.csv"
    summary_csv = out_dir / "unseen_summary_by_model.csv"

    write_csv(str(per_video_csv), model_rows, MODEL_ROW_COLUMNS)
    write_csv(str(consensus_csv), consensus_rows, CONSENSUS_COLUMNS)
    write_csv(str(summary_csv), summary_rows, SUMMARY_COLUMNS)

    plot_predicted_correctness(summary_rows, str(out_dir / "unseen_predicted_correctness.png"))
    plot_error_type_distribution(model_rows, str(out_dir / "unseen_error_type_distribution.png"))
    plot_consensus(consensus_rows, str(out_dir / "unseen_model_agreement.png"))

    run_meta_path = out_dir / "run_info.txt"
    with open(run_meta_path, "w", encoding="utf-8") as f:
        f.write("Unseen inference report\n")
        f.write(f"raw_dir={raw_dir}\n")
        f.write(f"features_dir={features_dir}\n")
        f.write(f"asformer_ckpt={asf_ckpt_path}\n")
        f.write(f"mstcn_ckpt={mst_ckpt_path}\n")
        f.write(f"videos={len(feat_files)}\n")
        f.write(f"fps={args.fps}\n")
        f.write(f"min_breath_hold_sec={args.min_breath_hold_sec}\n")
        f.write(f"mouth_open_threshold={args.mouth_open_threshold}\n")
        f.write(f"mouth_open_ratio_threshold={args.mouth_open_ratio_threshold}\n")
        f.write(f"mouth_open_delta_threshold={args.mouth_open_delta_threshold}\n")

    print("Done.")
    print(f"CSV: {per_video_csv}")
    print(f"CSV: {consensus_csv}")
    print(f"CSV: {summary_csv}")
    print(f"PNG: {out_dir / 'unseen_predicted_correctness.png'}")
    print(f"PNG: {out_dir / 'unseen_error_type_distribution.png'}")
    print(f"PNG: {out_dir / 'unseen_model_agreement.png'}")


if __name__ == "__main__":
    main()
