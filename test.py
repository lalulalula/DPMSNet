import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import TestSetLoader1
from metrics import PD_FA_new, ROCMetric, mIoU
from net import DPMSNetWrapper
from utils import seed_pytorch


MODEL_NAME = "DPMSNet"
LOSS_NAME = "AdaptiveSoftIoUFocalLoss"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DPMSNet")
    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="Root directory containing the IRSTD datasets",
    )
    parser.add_argument(
        "--dataset_names",
        default=["NUAA-SIRST"],
        nargs="+",
    )
    parser.add_argument(
        "--train_dataset",
        default="NUAA-SIRST",
        help="Dataset used to compute the normalization statistics",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a DPMSNet checkpoint")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--save_roc", action="store_true")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def irstd_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    masks = torch.stack([item[1] for item in batch])
    descriptions = [item[2] for item in batch]
    return images, masks, descriptions


def evaluate(model, args, dataset_name, device):
    dataset = TestSetLoader1(
        args.dataset_dir,
        args.train_dataset,
        dataset_name,
        img_norm_cfg=None,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=irstd_collate_fn,
    )

    iou_metric = mIoU()
    detection_metric = PD_FA_new()
    roc_metric = ROCMetric(1, 10)

    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device)
            predictions, _, _ = model(images, None)
            binary_prediction = predictions > args.threshold
            image_size = [images.size(2), images.size(3)]

            roc_metric.update(predictions, masks)
            iou_metric.update(binary_prediction.cpu(), masks.cpu())
            detection_metric.update(
                binary_prediction[0, 0].cpu(),
                masks[0, 0].cpu(),
                image_size,
            )

    pixel_accuracy, mean_iou = iou_metric.get()
    probability_detection, false_alarm, target_false_alarm = detection_metric.get()
    print(f"Dataset: {dataset_name}")
    print(f"Pixel accuracy: {pixel_accuracy:.6f}")
    print(f"mIoU: {mean_iou:.6f}")
    print(f"Pd: {probability_detection:.6f}")
    print(f"Fa: {false_alarm:.6e}")
    print(f"Fat: {target_false_alarm:.6f}")

    if args.save_roc:
        true_positive_rates, false_positive_rates, _ = roc_metric.get()
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        roc_path = output_dir / f"{dataset_name}_{MODEL_NAME}_ROC.txt"
        with roc_path.open("w", encoding="utf-8") as stream:
            for false_positive, true_positive in zip(
                false_positive_rates,
                true_positive_rates,
            ):
                stream.write(f"{false_positive:.6f} {true_positive:.6f}\n")
        print(f"ROC data saved to {roc_path}")


def main():
    args = parse_args()
    seed_pytorch(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DPMSNetWrapper(
        model_name=MODEL_NAME,
        mode="test",
        loss=LOSS_NAME,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    for dataset_name in args.dataset_names:
        evaluate(model, args, dataset_name, device)


if __name__ == "__main__":
    main()
