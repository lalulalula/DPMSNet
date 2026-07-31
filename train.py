import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import TestSetLoader1, TrainSetLoader2
from metrics import PD_FA_new, mIoU
from net import DPMSNetWrapper
from utils import seed_pytorch


MODEL_NAME = "DPMSNet"
LOSS_NAME = "AdaptiveSoftIoUFocalLoss"


def parse_args():
    parser = argparse.ArgumentParser(description="Train DPMSNet")
    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="Root directory containing the IRSTD datasets",
    )
    parser.add_argument(
        "--dataset_names",
        default=["NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K"],
        nargs="+",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--min_learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--eval_start", type=int, default=300)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--output_dir", default="./logs")
    parser.add_argument("--resume", default=None, help="Checkpoint used to resume training")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def irstd_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    masks = torch.stack([item[1] for item in batch])
    descriptions = [item[2] for item in batch]
    return images, masks, descriptions


def final_prediction(predictions):
    if isinstance(predictions, (tuple, list)):
        return predictions[-1]
    return predictions


def evaluate(model, args, dataset_name, device):
    dataset = TestSetLoader1(
        args.dataset_dir,
        dataset_name,
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
    losses = []
    model.eval()

    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device)
            predictions, _, _ = model(images, None)
            predictions = final_prediction(predictions)
            losses.append(model.loss(predictions, masks.to(device)).item())

            image_size = [images.size(2), images.size(3)]
            binary_prediction = predictions > args.threshold
            iou_metric.update(binary_prediction.cpu(), masks.cpu())
            detection_metric.update(
                binary_prediction[0, 0].cpu(),
                masks[0, 0].cpu(),
                image_size,
            )

    pixel_accuracy, mean_iou = iou_metric.get()
    probability_detection, false_alarm, target_false_alarm = detection_metric.get()
    return {
        "loss": float(np.mean(losses)),
        "pixel_accuracy": pixel_accuracy,
        "mean_iou": mean_iou,
        "probability_detection": probability_detection,
        "false_alarm": false_alarm,
        "target_false_alarm": target_false_alarm,
    }


def save_checkpoint(path, epoch, model, optimizer, scheduler, loss_history):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "loss_history": loss_history,
        },
        path,
    )


def train_dataset(args, dataset_name, device):
    train_set = TrainSetLoader2(
        dataset_dir=args.dataset_dir,
        dataset_name=dataset_name,
        patch_size=args.patch_size,
        img_norm_cfg=None,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=irstd_collate_fn,
    )

    model = DPMSNetWrapper(
        model_name=MODEL_NAME,
        mode="train",
        loss=LOSS_NAME,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_learning_rate,
    )

    start_epoch = 0
    loss_history = []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]
        loss_history = checkpoint.get("loss_history", [])

    run_name = f"{dataset_name}_{MODEL_NAME}_{LOSS_NAME}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    best_mean_iou = float("-inf")

    with log_path.open("w", encoding="utf-8") as log_file:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            detection_losses = []
            contrastive_losses = []

            for images, masks, descriptions in train_loader:
                if images.size(0) == 1:
                    continue

                images = images.to(device)
                masks = masks.to(device)
                predictions, image_embedding, text_embedding = model(images, descriptions)
                detection_loss = model.loss(predictions, masks)
                contrastive_loss = model.contrastive_loss(image_embedding, text_embedding)
                total_loss = detection_loss + contrastive_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                detection_losses.append(detection_loss.item())
                contrastive_losses.append(contrastive_loss.item())

            scheduler.step()
            epoch_losses = {
                "epoch": epoch + 1,
                "detection": float(np.mean(detection_losses)),
                "contrastive": float(np.mean(contrastive_losses)),
            }
            loss_history.append(epoch_losses)

            if (epoch + 1) % args.log_interval == 0:
                message = (
                    f"Epoch {epoch + 1:04d} | "
                    f"detection_loss={epoch_losses['detection']:.6f} | "
                    f"contrastive_loss={epoch_losses['contrastive']:.6f}"
                )
                print(message)
                log_file.write(message + "\n")
                log_file.flush()

            should_evaluate = (
                epoch + 1 >= args.eval_start
                and (epoch + 1) % args.eval_interval == 0
            )
            if should_evaluate:
                results = evaluate(model, args, dataset_name, device)
                message = (
                    f"Evaluation {epoch + 1:04d} | "
                    f"mIoU={results['mean_iou']:.6f} | "
                    f"Pd={results['probability_detection']:.6f} | "
                    f"Fa={results['false_alarm']:.6e} | "
                    f"Fat={results['target_false_alarm']:.6f}"
                )
                print(message)
                log_file.write(message + "\n")
                log_file.flush()

                if results["mean_iou"] > best_mean_iou:
                    best_mean_iou = results["mean_iou"]
                    save_checkpoint(
                        run_dir / "best.pth.tar",
                        epoch + 1,
                        model,
                        optimizer,
                        scheduler,
                        loss_history,
                    )

            if (epoch + 1) % args.save_interval == 0:
                save_checkpoint(
                    run_dir / f"epoch_{epoch + 1:04d}.pth.tar",
                    epoch + 1,
                    model,
                    optimizer,
                    scheduler,
                    loss_history,
                )


def main():
    args = parse_args()
    seed_pytorch(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for dataset_name in args.dataset_names:
        print(f"Training {MODEL_NAME} on {dataset_name}")
        train_dataset(args, dataset_name, device)


if __name__ == "__main__":
    main()
