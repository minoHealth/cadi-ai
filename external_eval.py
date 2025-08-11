"""Unified evaluation & diagnostics script for CADI AI YOLO models.

Refactored to integrate with the training notebook pipeline and support:
 - Environment auto-detection (Kaggle / Colab / Local)
 - Auto discovery of latest trained weights (runs/**/weights/{best,last}.pt)
 - Data / working directory resolution via config.yaml + heuristics
 - Dataset balance analysis (counts + size distributions) with resumable checkpoints
 - Per-class detection diagnostics for problematic / rare classes
 - Optional automatic selection of lowest-frequency classes when not provided
 - Anchor vs object size visualization (if anchors are available)
 - Consolidated report + plots under WORKING_DIR/eval/<timestamp>

CLI example:
  python external_eval.py --config config.yaml --problematic "abiotic,disease" --conf 0.2

Keep code <350 lines (approx) for readability.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from tqdm import tqdm
from ultralytics import YOLO

# ---------------------------- Utility & Environment ---------------------------- #

def log(msg: str) -> None:
    print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] {msg}")


def detect_environment() -> str:
    cwd = os.getcwd().replace('\\', '/').lower()
    if '/kaggle' in cwd:
        return 'kaggle'
    if '/content' in cwd:
        return 'colab'
    return 'local'


def load_config(config_path: Path) -> Dict:
    if not config_path.exists():
        log(f"Config file not found at {config_path}, proceeding with defaults.")
        return {}
    with config_path.open('r') as f:
        return yaml.safe_load(f) or {}


def resolve_paths(cfg: Dict, env: str) -> Tuple[Path, Path, Path]:
    # PROJECT_DIR: current repo root (file's parent) fallback to cwd
    project_dir = Path(cfg.get('project_dir') or Path.cwd()).resolve()

    # WORKING_DIR: training outputs directory
    working_dir = Path(
        cfg.get('working_dir')
        or cfg.get('output_dir')  # allow legacy naming
        or (project_dir / 'training_outputs')
    ).resolve()
    working_dir.mkdir(parents=True, exist_ok=True)

    # DATASET_PATH (root) is optional; often in data.yaml already. Keep for reference.
    dataset_root = Path(cfg.get('dataset_root') or project_dir / 'dataset').resolve()
    return project_dir, working_dir, dataset_root


def find_data_yaml(project_dir: Path, working_dir: Path, cfg: Dict) -> Path | None:
    explicit = cfg.get('data') or cfg.get('data_yaml') or cfg.get('data_path')
    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates.extend([
        project_dir / 'data.yaml',
        working_dir / 'data.yaml',
        Path('data.yaml'),
    ])
    for c in candidates:
        p = Path(c).expanduser().resolve() if not isinstance(c, Path) else c
        if p.exists():
            return p
    return None


def find_latest_weights(project_dir: Path, explicit: str | None = None) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        return p if p.exists() else None
    patterns = [
        str(project_dir / 'runs' / '**' / 'weights' / 'best.pt'),
        str(project_dir / 'runs' / '**' / 'weights' / 'last.pt'),
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(glob(pat, recursive=True))
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda p: os.path.getmtime(p), reverse=True)
    return Path(candidates[0]).resolve()


def read_yaml(path: Path) -> Dict:
    with path.open('r') as f:
        return yaml.safe_load(f) or {}


# ---------------------------- Dataset Analysis ---------------------------- #

def gather_label_files(data_cfg: Dict, base: Path) -> List[Path]:
    train_path = data_cfg.get('train')
    if not train_path:
        return []
    train_path = Path(train_path)
    if not train_path.is_absolute():
        train_path = (base / train_path).resolve()
    # Heuristic: if points to images folder or parent, locate labels sibling
    if train_path.is_dir():
        if train_path.name == 'images':
            labels_dir = train_path.parent / 'labels'
        else:
            # try subfolder labels
            labels_dir = train_path / 'labels'
        if not labels_dir.exists():
            # fallback: search recursively for *.txt under something named labels
            txts = list(train_path.rglob('*.txt'))
            return txts
        return list(labels_dir.rglob('*.txt'))
    return []


def dataset_analysis(data_yaml: Path, output_dir: Path, resume: bool = True, checkpoint_every: int = 1000) -> Tuple[Dict[str, int], Dict[str, List[float]]]:
    cfg = read_yaml(data_yaml)
    class_names = cfg.get('names') or []
    label_files = gather_label_files(cfg, data_yaml.parent)
    class_counts = {n: 0 for n in class_names}
    class_sizes = {n: [] for n in class_names}

    ckpt_path = output_dir / 'dataset_analysis_checkpoint.json'
    start_idx = 0
    if resume and ckpt_path.exists():
        try:
            data = json.loads(ckpt_path.read_text())
            if set(data.get('class_counts', {}).keys()) == set(class_counts.keys()):
                class_counts.update(data['class_counts'])
                for k in class_sizes:
                    class_sizes[k] = data['class_sizes'].get(k, [])
                start_idx = data.get('processed_files', 0)
                log(f"Resuming dataset analysis at file index {start_idx}.")
        except Exception as e:
            log(f"Failed to load dataset analysis checkpoint: {e}")

    log(f"Scanning {len(label_files)} label files for class statistics...")
    for i, label_file in enumerate(tqdm(label_files[start_idx:], disable=not label_files)):
        try:
            lines = Path(label_file).read_text().strip().splitlines()
        except Exception:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                cid = int(parts[0]) if parts[0].isdigit() else None
                if cid is None or cid >= len(class_names):
                    continue
                cname = class_names[cid]
                w, h = float(parts[3]), float(parts[4])
                area = w * h  # normalized area
                class_counts[cname] += 1
                class_sizes[cname].append(area)
        if (i + 1) % checkpoint_every == 0:
            _save_dataset_ckpt(ckpt_path, class_counts, class_sizes, start_idx + i + 1)
    _save_dataset_ckpt(ckpt_path, class_counts, class_sizes, start_idx + len(label_files[start_idx:]))

    _plot_class_distribution(class_counts, output_dir)
    _plot_size_distribution(class_sizes, output_dir)
    _write_class_summary(class_counts, class_sizes, output_dir)
    return class_counts, class_sizes


def _save_dataset_ckpt(path: Path, counts: Dict[str, int], sizes: Dict[str, List[float]], processed: int):
    data = {
        'class_counts': counts,
        'class_sizes': sizes,
        'processed_files': processed,
    }
    path.write_text(json.dumps(data))


def _plot_class_distribution(class_counts: Dict[str, int], out_dir: Path):
    if not class_counts:
        return
    names = list(class_counts.keys())
    vals = [class_counts[n] for n in names]
    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, vals)
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width() / 2, h + max(vals) * 0.01 + 0.1, str(h), ha='center', va='bottom', fontsize=8)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Instances')
    plt.title('Class Distribution')
    plt.tight_layout()
    plt.savefig(out_dir / 'class_distribution.png')
    plt.close()


def _plot_size_distribution(class_sizes: Dict[str, List[float]], out_dir: Path):
    data = []
    for k, arr in class_sizes.items():
        if arr:
            data.extend([(k, a * 100) for a in arr])
    if not data:
        return
    df = pd.DataFrame(data, columns=['Class', 'Size (% area)'])
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Class', y='Size (% area)', data=df, showfliers=False)
    plt.xticks(rotation=45, ha='right')
    plt.title('Normalized Object Size Distribution')
    plt.tight_layout()
    plt.savefig(out_dir / 'size_distribution.png')
    plt.close()


def _write_class_summary(class_counts: Dict[str, int], class_sizes: Dict[str, List[float]], out_dir: Path):
    rows = []
    for k in class_counts:
        arr = class_sizes[k]
        rows.append({
            'Class': k,
            'Count': class_counts[k],
            'Avg Size %': (np.mean(arr) * 100) if arr else 0,
            'Min Size %': (np.min(arr) * 100) if arr else 0,
            'Max Size %': (np.max(arr) * 100) if arr else 0,
        })
    pd.DataFrame(rows).to_csv(out_dir / 'class_summary.csv', index=False)


# ---------------------------- Prediction Diagnostics ---------------------------- #

def prediction_diagnostics(
    model: YOLO,
    data_yaml: Path,
    target_classes: List[str],
    output_dir: Path,
    conf: float = 0.2,
    limit: int | None = None,
    min_iou: float = 0.5,
    overlap_thresh: float = 0.3,
    resume: bool = True,
    batch: int = 16,
    external_val: Path | None = None,
) -> Dict[str, float]:
    data_cfg = read_yaml(data_yaml)
    class_names = data_cfg.get('names') or []
    idx_lookup = {c: i for i, c in enumerate(class_names)}
    indices = [idx_lookup[c] for c in target_classes if c in idx_lookup]

    # Run a validation pass to produce confusion matrix, etc.
    try:
        log("Running model.val(...) for confusion matrix generation...")
        results = model.val(data=str(data_yaml), batch=batch, conf=conf, save_json=True, verbose=False)
        if hasattr(results, 'confusion_matrix') and results.confusion_matrix is not None:
            try:
                cm = results.confusion_matrix.matrix
                _plot_confusion_matrix(cm, class_names, output_dir)
            except Exception as e:
                log(f"Could not plot confusion matrix: {e}")
    except Exception as e:
        log(f"Validation step failed (continuing): {e}")

    # Determine validation source (external overrides data.yaml val)
    if external_val is not None:
        val_root = external_val
        log(f"Using external validation path: {val_root}")
    else:
        val_path = data_cfg.get('val')
        if not val_path:
            log('No val path in data.yaml; skipping detection diagnostics.')
            return {}
        val_root = Path(val_path)
        if not val_root.is_absolute():
            val_root = (data_yaml.parent / val_root).resolve()

    image_files: List[Path] = []
    if val_root.is_dir():
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            image_files.extend(val_root.rglob(ext))
    elif val_root.suffix == '.txt':  # list file
        try:
            lines = val_root.read_text().strip().splitlines()
            image_files = [Path(l.strip()) for l in lines if l.strip()]
        except Exception:
            pass
    if limit and len(image_files) > limit:
        image_files = image_files[:limit]

    # Build label paths
    label_files = [
        Path(str(p).replace('/images/', '/labels/').replace('\\images\\', '\\labels\\'))
        .with_suffix('.txt') for p in image_files
    ]

    for cls in target_classes:
        (output_dir / f'missed_{cls}').mkdir(parents=True, exist_ok=True)

    success = {c: 0 for c in target_classes}
    total = {c: 0 for c in target_classes}
    ckpt_path = output_dir / 'prediction_checkpoint.json'
    start_idx = 0
    if resume and ckpt_path.exists():
        try:
            data = json.loads(ckpt_path.read_text())
            if set(data.get('success_count', {}).keys()) == set(success.keys()):
                success.update(data['success_count'])
                total.update(data['total_count'])
                start_idx = data.get('processed_images', 0)
                log(f"Resuming prediction diagnostics at image index {start_idx}.")
        except Exception:
            pass

    for i, (img_path, lbl_path) in enumerate(tqdm(list(zip(image_files, label_files))[start_idx:], disable=not image_files)):
        if not lbl_path.exists():
            continue
        try:
            gt_lines = lbl_path.read_text().strip().splitlines()
        except Exception:
            continue
        gt_objs = []
        for line in gt_lines:
            ps = line.split()
            if len(ps) >= 5 and ps[0].isdigit():
                cid = int(ps[0])
                if cid < len(class_names):
                    x, y, w, h = map(float, ps[1:5])
                    gt_objs.append({'cid': cid, 'bbox': (x, y, w, h)})
        if not any(o['cid'] in indices for o in gt_objs):
            continue

        # run inference
        try:
            pred_res = model(str(img_path), conf=conf, verbose=False)[0]
            pred_boxes = pred_res.boxes.data.cpu().numpy() if hasattr(pred_res.boxes, 'data') else []
        except Exception:
            pred_boxes = []

        # load image only if needed for saving misses
        img = None
        for gt in gt_objs:
            if gt['cid'] not in indices:
                continue
            cname = class_names[gt['cid']]
            total[cname] += 1
            x, y, w, h = gt['bbox']
            # convert normalized to pixel for IoU calculations with predictions (pred already in pixel)
            if img is None:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                h_img, w_img = img.shape[:2]
            x1 = int((x - w / 2) * w_img)
            y1 = int((y - h / 2) * h_img)
            x2 = int((x + w / 2) * w_img)
            y2 = int((y + h / 2) * h_img)
            detected = False
            for box in pred_boxes:
                px1, py1, px2, py2, pconf, pcl = box
                # IoU
                xi1, yi1 = max(x1, int(px1)), max(y1, int(py1))
                xi2, yi2 = min(x2, int(px2)), min(y2, int(py2))
                inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                area_gt = (x2 - x1) * (y2 - y1)
                area_pred = (int(px2) - int(px1)) * (int(py2) - int(py1))
                union = area_gt + area_pred - inter
                iou = inter / union if union > 0 else 0
                if iou >= min_iou and int(pcl) == gt['cid']:
                    success[cname] += 1
                    detected = True
                    break
            if not detected:
                # visualize miss
                overlay = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(overlay, f"GT {cname}", (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                for box in pred_boxes:
                    px1, py1, px2, py2, pconf, pcl = box
                    # overlap threshold proportion of gt box
                    ov_x1, ov_y1 = max(x1, int(px1)), max(y1, int(py1))
                    ov_x2, ov_y2 = min(x2, int(px2)), min(y2, int(py2))
                    ov_area = max(0, ov_x2 - ov_x1) * max(0, ov_y2 - ov_y1)
                    if ov_area / ((x2 - x1) * (y2 - y1) + 1e-9) >= overlap_thresh:
                        pcname = class_names[int(pcl)] if int(pcl) < len(class_names) else 'UNK'
                        cv2.rectangle(overlay, (int(px1), int(py1)), (int(px2), int(py2)), (255, 0, 0), 1)
                        cv2.putText(overlay, f"{pcname}:{pconf:.2f}", (int(px1), max(0, int(py1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                out_path = output_dir / f"missed_{cname}" / img_path.name
                plt.figure(figsize=(6, 6))
                plt.imshow(overlay)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(out_path)
                plt.close()
        if (i + 1) % 50 == 0:
            _save_prediction_ckpt(ckpt_path, success, total, start_idx + i + 1)
    _save_prediction_ckpt(ckpt_path, success, total, start_idx + len(image_files[start_idx:]))

    # detection rates
    rates = {c: (success[c] / total[c]) if total[c] else 0.0 for c in target_classes}
    _plot_detection_rates(rates, output_dir, conf)
    _write_detection_summary(rates, success, total, output_dir)
    return rates


def _save_prediction_ckpt(path: Path, success: Dict[str, int], total: Dict[str, int], processed: int):
    path.write_text(json.dumps({
        'success_count': success,
        'total_count': total,
        'processed_images': processed,
    }))


def _plot_detection_rates(rates: Dict[str, float], out_dir: Path, conf: float):
    if not rates:
        return
    plt.figure(figsize=(8, 5))
    names = list(rates.keys())
    vals = [rates[k] for k in names]
    bars = plt.bar(names, vals)
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width()/2, h + 0.02, f"{h:.2f}", ha='center', va='bottom')
    plt.ylim(0, 1)
    plt.ylabel('Detection Rate')
    plt.title(f'Detection Rates (conf={conf})')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out_dir / 'detection_rates.png')
    plt.close()


def _write_detection_summary(rates: Dict[str, float], success: Dict[str, int], total: Dict[str, int], out_dir: Path):
    rows = []
    for c in rates:
        rows.append({'Class': c, 'Total': total[c], 'Detected': success[c], 'Detection Rate': rates[c]})
    pd.DataFrame(rows).to_csv(out_dir / 'detection_summary.csv', index=False)


# ---------------------------- Anchor Analysis ---------------------------- #

def anchor_analysis(data_yaml: Path, class_sizes: Dict[str, List[float]], out_dir: Path):
    model_yaml = data_yaml.parent / 'model.yaml'
    if not model_yaml.exists():
        log('No model.yaml (anchors) found; skipping anchor analysis.')
        return
    try:
        cfg = read_yaml(model_yaml)
        anchors = cfg.get('anchors')
        if not anchors:
            log('No anchors field in model.yaml; skipping.')
            return
    except Exception as e:
        log(f'Failed reading model.yaml: {e}')
        return
    plt.figure(figsize=(10, 6))
    # scatter object sizes (approx width/height via sqrt(area))
    for cls, arr in class_sizes.items():
        if not arr:
            continue
        sample = arr if len(arr) <= 4000 else arr[:4000]
        side = np.sqrt(sample)
        plt.scatter(side, side, s=4, alpha=0.15, label=cls)
    aw, ah = [], []
    for grp in anchors:
        for w, h in grp:
            aw.append(w)
            ah.append(h)
    plt.scatter(aw, ah, c='red', marker='x', s=80, label='Anchors')
    plt.xlabel('Width (normalized)')
    plt.ylabel('Height (normalized)')
    plt.title('Object Sizes vs Anchors (approx square assumption)')
    plt.legend(loc='upper right', fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / 'anchor_analysis.png')
    plt.close()


def _plot_confusion_matrix(matrix, class_names: List[str], out_dir: Path):
    ext_names = class_names + ['Background'] if matrix.shape[0] == len(class_names) + 1 else class_names
    plt.figure(figsize=(12, 9))
    sns.heatmap(matrix, annot=False, fmt='g', xticklabels=ext_names, yticklabels=ext_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(out_dir / 'confusion_matrix.png')
    plt.close()


# ---------------------------- Recommendations ---------------------------- #

def generate_recommendations(problematic: List[str], counts: Dict[str, int], rates: Dict[str, float]) -> List[str]:
    recs = []
    for cls in problematic:
        ccount = counts.get(cls, 0)
        rate = rates.get(cls, 0.0)
        if ccount < 100:
            recs.append(f"Class '{cls}' has few samples ({ccount}); consider collecting more data / augmentation.")
        if rate < 0.2:
            recs.append(f"Class '{cls}' detection rate {rate:.2f} is low at relaxed conf; inspect annotation quality / class imbalance.")
        if 0.2 <= rate < 0.5:
            recs.append(f"Class '{cls}' moderate detection rate {rate:.2f}; consider targeted augmentation or adjusting loss (e.g., Focal).")
    if any(r > 0.3 for r in rates.values()):
        recs.append("Some classes appear at low confidence but may be filtered at higher thresholds; consider per-class thresholding.")
    return recs


# ---------------------------- Main Orchestration ---------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description='CADI AI YOLO evaluation & diagnostics')
    p.add_argument('--config', default='config.yaml', help='Path to config.yaml')
    p.add_argument('--weights', default=None, help='Explicit weights .pt path (override auto)')
    p.add_argument('--problematic', default=None, help='Comma-separated problematic class names')
    p.add_argument('--auto-problematic-k', type=int, default=2, help='If --problematic not set, select K lowest-frequency classes')
    p.add_argument('--conf', type=float, default=0.2, help='Confidence threshold for diagnostics inference & val')
    p.add_argument('--limit', type=int, default=None, help='Limit number of validation images processed')
    p.add_argument('--batch', type=int, default=16, help='Batch size for model.val')
    p.add_argument('--min-iou', type=float, default=0.5, help='IoU threshold for true positive')
    p.add_argument('--overlap-thresh', type=float, default=0.3, help='Overlap required (fraction of GT) to visualize near-miss predictions')
    p.add_argument('--no-resume', action='store_true', help='Disable checkpoint resume')
    p.add_argument('--output-dir', default=None, help='Explicit output directory (otherwise auto under WORKING_DIR/eval)')
    p.add_argument('--external-val', default=None, help='External validation path (directory or list file) used only for diagnostics; does NOT modify data.yaml')
    return p.parse_args()


def main():
    args = parse_args()
    env = detect_environment()
    cfg = load_config(Path(args.config))
    project_dir, working_dir, dataset_root = resolve_paths(cfg, env)
    resume = not args.no_resume

    log(f"Environment: {env}")
    log(f"Project dir: {project_dir}")
    log(f"Working dir: {working_dir}")

    data_yaml = find_data_yaml(project_dir, working_dir, cfg)
    if not data_yaml:
        log('ERROR: data.yaml not found. Ensure dataset preparation step completed.')
        sys.exit(2)
    log(f"Using data.yaml: {data_yaml}")

    weights_path = find_latest_weights(project_dir, args.weights)
    if not weights_path:
        log('ERROR: No weights file found (runs/**/weights/(best|last).pt). Train model first.')
        sys.exit(3)
    log(f"Using weights: {weights_path}")

    # Prepare output dir
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) if args.output_dir else (working_dir / 'eval' / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {output_dir}")

    # Load model
    model = YOLO(str(weights_path))
    log('Model loaded.')

    # Dataset analysis
    class_counts, class_sizes = dataset_analysis(data_yaml, output_dir, resume=resume)
    log('Dataset analysis complete.')

    data_cfg = read_yaml(data_yaml)
    all_classes = data_cfg.get('names') or []

    if args.problematic:
        problematic = [c.strip() for c in args.problematic.split(',') if c.strip() in all_classes]
    else:
        # auto-pick rarest K classes
        ranked = sorted(all_classes, key=lambda c: class_counts.get(c, 0))
        problematic = ranked[: max(1, args.auto_problematic_k)]
    if not problematic:
        log('No valid problematic classes specified or found; aborting diagnostics step.')
        rates = {}
    else:
        log(f"Problematic classes: {problematic}")
        external_val_path = Path(args.external_val).resolve() if args.external_val else None
        if external_val_path and not external_val_path.exists():
            log(f"WARNING: external validation path {external_val_path} does not exist; falling back to data.yaml val")
            external_val_path = None
        rates = prediction_diagnostics(
            model,
            data_yaml,
            problematic,
            output_dir,
            conf=args.conf,
            limit=args.limit,
            min_iou=args.min_iou,
            overlap_thresh=args.overlap_thresh,
            resume=resume,
            batch=args.batch,
            external_val=external_val_path,
        )
        log('Prediction diagnostics complete.')

    # Anchor analysis
    anchor_analysis(data_yaml, class_sizes, output_dir)

    # Recommendations & report
    recs = generate_recommendations(problematic, class_counts, rates) if problematic else []
    report = {
        'timestamp_utc': timestamp,
        'environment': env,
        'project_dir': str(project_dir),
        'working_dir': str(working_dir),
        'data_yaml': str(data_yaml),
        'weights': str(weights_path),
        'classes': all_classes,
        'problematic_classes': problematic,
        'class_counts': class_counts,
        'detection_rates': rates,
        'recommendations': recs,
        'args': vars(args),
        'external_val_used': args.external_val is not None,
        'external_val_path': args.external_val,
    }
    (output_dir / 'analysis_report.json').write_text(json.dumps(report, indent=2))
    log('Report written (analysis_report.json).')

    log('===== Summary =====')
    log(f"Problematic classes: {problematic}")
    log(f"Detection rates: {rates}")
    if recs:
        log('Recommendations:')
        for r in recs:
            log(f" - {r}")
    log(f"Artifacts saved under: {output_dir}")


if __name__ == '__main__':
    main()