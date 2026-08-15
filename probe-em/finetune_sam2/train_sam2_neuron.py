"""Fine-tune SAM 2 for neuron segmentation prompts.

This script fine-tunes the SAM 2 prompt encoder and mask decoder while keeping
the image encoder frozen. It expects a prepared 2D training set:

    data_root/
      imgs/
        sample_001_raw.tif
      masks/
        sample_001_label.tif

Mask files should contain integer instance labels, where 0 is background.
Install the official SAM 2 package first:
https://github.com/facebookresearch/sam2
"""

import argparse
import os
import random
import time
from pathlib import Path


cv2 = None
np = None
tifffile = None
torch = None


def load_training_dependencies():
    """Load heavy training dependencies after argument parsing."""
    global cv2, np, tifffile, torch
    try:
        import cv2 as cv2_module
        import numpy as np_module
        import tifffile as tifffile_module
        import torch as torch_module
    except ImportError as exc:
        raise ImportError(
            "Training dependencies are missing. Install the repository "
            "requirements before running fine-tuning."
        ) from exc

    cv2 = cv2_module
    np = np_module
    tifffile = tifffile_module
    torch = torch_module


def dice_loss(pred_mask, gt_mask, eps=1e-6):
    """Dice loss for binary mask prediction."""
    pred_flat = pred_mask.reshape(pred_mask.shape[0], -1)
    gt_flat = gt_mask.reshape(gt_mask.shape[0], -1)
    intersection = (pred_flat * gt_flat).sum(1)
    union = pred_flat.sum(1) + gt_flat.sum(1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def focal_loss(pred_mask, gt_mask, alpha=0.75, gamma=2.0, eps=1e-6):
    """Focal loss for foreground-background imbalance."""
    loss = -(
        alpha * gt_mask * ((1 - pred_mask) ** gamma) * torch.log(pred_mask + eps)
        + (1 - alpha) * (1 - gt_mask) * (pred_mask ** gamma) * torch.log(1 - pred_mask + eps)
    )
    return loss.mean()


def get_largest_component(binary_mask, min_area_threshold=100):
    """Return the largest connected component if it is large enough."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels < 2:
        return None, False

    max_label_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    max_area = stats[max_label_idx, cv2.CC_STAT_AREA]
    if max_area < min_area_threshold:
        return None, False

    return (labels == max_label_idx).astype(np.uint8), True


def get_safe_sampling_area(mask, margin=3):
    """Erode the object mask to sample prompts away from instance boundaries."""
    kernel_size = 2 * margin + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    safe_mask = cv2.erode(mask, kernel, iterations=1)
    if cv2.countNonZero(safe_mask) == 0:
        return mask
    return safe_mask


def get_training_sample_n_area(img_path, mask_path, area_num, min_pixel_area=100, safety_margin=3):
    """Sample up to ``area_num`` valid instances and one positive point per instance."""
    try:
        mask_all = tifffile.imread(mask_path)
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return None

    instance_ids = np.unique(mask_all)
    instance_ids = instance_ids[instance_ids != 0]
    if len(instance_ids) == 0:
        return None

    np.random.shuffle(instance_ids)
    collected_masks = []
    collected_points = []
    collected_labels = []

    for target_id in instance_ids:
        if len(collected_masks) >= area_num:
            break

        raw_binary_mask = (mask_all == target_id).astype(np.uint8)
        clean_mask, is_valid = get_largest_component(
            raw_binary_mask, min_area_threshold=min_pixel_area
        )
        if not is_valid:
            continue

        safe_pos_mask = get_safe_sampling_area(clean_mask, margin=safety_margin)
        pos_coords = np.argwhere(safe_pos_mask > 0)
        if len(pos_coords) == 0:
            continue

        yx = pos_coords[np.random.choice(len(pos_coords))]
        point_xy = [int(yx[1]), int(yx[0])]

        collected_masks.append(clean_mask)
        collected_points.append([point_xy])
        collected_labels.append([1])

    if len(collected_masks) == 0:
        return None

    masks_np = np.array(collected_masks, dtype=np.float32)
    points_np = np.array(collected_points, dtype=np.float32)
    labels_np = np.array(collected_labels, dtype=np.float32)
    return img, masks_np, points_np, labels_np


def discover_training_pairs(data_root, image_dir="imgs", mask_dir="masks",
                            raw_suffix="_raw.tif", label_suffix="_label.tif"):
    """Find image-mask pairs in a prepared 2D training dataset."""
    img_dir = Path(data_root) / image_dir
    label_dir = Path(data_root) / mask_dir
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {label_dir}")

    pairs = []
    for img_path in sorted(img_dir.glob(f"*{raw_suffix}")):
        mask_name = img_path.name.replace(raw_suffix, label_suffix)
        mask_path = label_dir / mask_name
        if mask_path.exists():
            pairs.append((str(img_path), str(mask_path)))

    if not pairs:
        raise ValueError(
            f"No training pairs found under {data_root}. "
            f"Expected images like '*{raw_suffix}' and masks like '*{label_suffix}'."
        )
    return pairs


def maybe_init_wandb(args):
    """Initialize W&B only when explicitly requested."""
    if not args.use_wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("Install wandb or remove --use-wandb.") from exc

    run_name = args.wandb_run_name
    if run_name is None:
        run_name = f"sam2_neuron_{time.strftime('%Y%m%d_%H%M%S')}"

    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "max_iterations": args.max_iterations,
            "area_num": args.area_num,
            "min_pixel_area": args.min_pixel_area,
            "safety_margin": args.safety_margin,
        },
    )
    return wandb


def configure_trainable_parameters(model, train_adapters=False):
    """Freeze the image encoder and train the prompt encoder plus mask decoder."""
    for param in model.parameters():
        param.requires_grad = False

    for param in model.sam_mask_decoder.parameters():
        param.requires_grad = True
    for param in model.sam_prompt_encoder.parameters():
        param.requires_grad = True

    if train_adapters:
        for name, param in model.named_parameters():
            if "adapter" in name.lower() or "adapternorm" in name.lower():
                param.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return trainable_params


def train_sam2(predictor, optimizer, scaler, data_list, args, wandb_run=None):
    """Run SAM 2 fine-tuning with gradient accumulation."""
    os.makedirs(args.output_dir, exist_ok=True)
    device = predictor.device
    log_file = Path(args.output_dir) / "training_log.txt"

    print("Starting SAM 2 fine-tuning")
    print(f"Training pairs: {len(data_list)}")
    print(f"Iterations: {args.max_iterations}")
    print(f"Batch size via gradient accumulation: {args.batch_size}")
    print(f"Output directory: {args.output_dir}")

    optimizer.zero_grad()
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("iteration\ttotal_loss\tfocal_loss\tdice_loss\tscore_loss\n")

    for iteration in range(1, args.max_iterations + 1):
        loss_tracker = {"total": 0.0, "seg": 0.0, "dice": 0.0, "score": 0.0}
        samples_processed = 0
        failed_trials = 0

        while samples_processed < args.batch_size:
            result = None
            while result is None:
                img_path, mask_path = random.choice(data_list)
                result = get_training_sample_n_area(
                    img_path,
                    mask_path,
                    area_num=args.area_num,
                    min_pixel_area=args.min_pixel_area,
                    safety_margin=args.safety_margin,
                )
                failed_trials += 1
                if failed_trials > args.max_sampling_trials:
                    raise RuntimeError(
                        "Failed to sample a valid training instance. "
                        "Check your masks or lower --min-pixel-area."
                    )

            image, gt_masks_np, input_points_np, input_labels_np = result
            gt_masks = torch.from_numpy(gt_masks_np).float().to(device)

            with torch.amp.autocast(device.type, enabled=(device.type in ("cuda", "mps"))):
                predictor.set_image(image)
                mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                    input_points_np,
                    input_labels_np,
                    box=None,
                    mask_logits=None,
                    normalize_coords=True,
                )
                sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                    points=(unnorm_coords, labels),
                    boxes=None,
                    masks=None,
                )

                high_res_features = [
                    feat_level[-1].unsqueeze(0)
                    for feat_level in predictor._features["high_res_feats"]
                ]

                low_res_masks, pred_scores, _, _ = predictor.model.sam_mask_decoder(
                    image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                    image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True,
                    repeat_image=(unnorm_coords.shape[0] > 1),
                    high_res_features=high_res_features,
                )

                pred_masks_high_res = predictor._transforms.postprocess_masks(
                    low_res_masks, predictor._orig_hw[-1]
                )
                pred_mask_prob = torch.sigmoid(pred_masks_high_res[:, 0])

                loss_seg = focal_loss(pred_mask_prob, gt_masks)
                loss_dice = dice_loss(pred_mask_prob, gt_masks)

                binary_pred = pred_mask_prob > 0.5
                inter = (gt_masks * binary_pred).sum(1).sum(1)
                union = gt_masks.sum(1).sum(1) + binary_pred.sum(1).sum(1) - inter
                iou_true = inter / (union + 1e-6)
                loss_score = torch.abs(pred_scores[:, 0] - iou_true).mean()

                total_loss = (
                    args.focal_weight * loss_seg
                    + args.dice_weight * loss_dice
                    + args.score_weight * loss_score
                )
                loss_normalized = total_loss / args.batch_size

            if scaler is not None:
                scaler.scale(loss_normalized).backward()
            else:
                loss_normalized.backward()

            loss_tracker["total"] += float(total_loss.item())
            loss_tracker["seg"] += float(loss_seg.item())
            loss_tracker["dice"] += float(loss_dice.item())
            loss_tracker["score"] += float(loss_score.item())
            samples_processed += 1

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

        if iteration % args.log_every == 0 or iteration == 1:
            avg_total = loss_tracker["total"] / args.batch_size
            avg_seg = loss_tracker["seg"] / args.batch_size
            avg_dice = loss_tracker["dice"] / args.batch_size
            avg_score = loss_tracker["score"] / args.batch_size

            print(
                f"Iter {iteration}/{args.max_iterations} | "
                f"Total: {avg_total:.4f} | "
                f"Focal: {avg_seg:.4f} | "
                f"Dice: {avg_dice:.4f} | "
                f"Score: {avg_score:.4f}"
            )

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(
                    f"{iteration}\t{avg_total:.6f}\t{avg_seg:.6f}\t"
                    f"{avg_dice:.6f}\t{avg_score:.6f}\n"
                )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "Loss/Total": avg_total,
                        "Loss/Focal": avg_seg,
                        "Loss/Dice": avg_dice,
                        "Loss/Score": avg_score,
                    },
                    step=iteration,
                )

        if iteration % args.save_every == 0 or iteration == args.max_iterations:
            save_path = Path(args.output_dir) / f"sam2_iter_{iteration}.pt"
            torch.save(predictor.model.state_dict(), save_path)
            print(f"Saved checkpoint: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune SAM 2 for neuron segmentation prompts.")
    parser.add_argument("--sam2-checkpoint", required=True, help="Path to a SAM 2 checkpoint.")
    parser.add_argument(
        "--model-cfg",
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="SAM 2 model config path, relative to the SAM 2 package if applicable.",
    )
    parser.add_argument("--data-root", required=True, help="Prepared 2D dataset root.")
    parser.add_argument("--image-dir", default="imgs", help="Image subdirectory under data root.")
    parser.add_argument("--mask-dir", default="masks", help="Mask subdirectory under data root.")
    parser.add_argument("--raw-suffix", default="_raw.tif", help="Suffix for raw image files.")
    parser.add_argument("--label-suffix", default="_label.tif", help="Suffix for label mask files.")
    parser.add_argument("--output-dir", default="checkpoints_finetuned", help="Checkpoint output directory.")
    parser.add_argument("--device", default=None, help="Device, e.g. cuda:0 or cpu. Default: auto.")

    parser.add_argument("--max-iterations", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=16, help="Effective batch size via accumulation.")
    parser.add_argument("--area-num", type=int, default=1, help="Maximum number of instances sampled per image.")
    parser.add_argument("--min-pixel-area", type=int, default=100)
    parser.add_argument("--safety-margin", type=int, default=3)
    parser.add_argument("--max-sampling-trials", type=int, default=1000)

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=4e-5)
    parser.add_argument("--focal-weight", type=float, default=0.9)
    parser.add_argument("--dice-weight", type=float, default=0.1)
    parser.add_argument("--score-weight", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-adapters", action="store_true")

    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default="sam2-neuron-finetune")
    parser.add_argument("--wandb-run-name", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    load_training_dependencies()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from probe_em.device import resolve_device, sam2_config_name
    device = torch.device(resolve_device(args.device))

    data_list = discover_training_pairs(
        args.data_root,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        raw_suffix=args.raw_suffix,
        label_suffix=args.label_suffix,
    )

    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        raise ImportError(
            "SAM 2 is required for fine-tuning. Install it from "
            "https://github.com/facebookresearch/sam2 before running this script."
        ) from exc

    print(f"Loading SAM 2 on {device}...")
    sam2_model = build_sam2(sam2_config_name(args.model_cfg), args.sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    trainable_params = configure_trainable_parameters(
        predictor.model,
        train_adapters=args.train_adapters,
    )
    print(f"Trainable tensors: {len(trainable_params)}")

    optimizer = torch.optim.AdamW(
        params=trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda")) if device.type == "cuda" else None
    wandb_run = maybe_init_wandb(args)

    start_time = time.time()
    train_sam2(
        predictor=predictor,
        optimizer=optimizer,
        scaler=scaler,
        data_list=data_list,
        args=args,
        wandb_run=wandb_run,
    )
    if wandb_run is not None:
        wandb_run.finish()
    print(f"Total runtime: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
