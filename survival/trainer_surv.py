import argparse
import pickle
import time

import pytorch_lightning as pl
import torch
from datamodule_surv import TabularDataModule
from engine_surv import EngineAttn
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

# Training settings
parser = argparse.ArgumentParser(
    description="Configurations for survival analysis training"
)
parser.add_argument("--data_dir", type=str, default=None, help="Input data directory")
parser.add_argument(
    "--time_bins_fp",
    type=str,
    default="/home/asohn3/baraslab/germline_somatic/Data/final_surv/mtlr_lbls.pkl",
    help="OS tgt bins for survival analysis",
)
parser.add_argument("--output_dir", type=str, default=None, help="Model results path")
# parser.add_argument(
#     '--device_ids',
#     type=int, nargs='+', default=None, help='Indices of which GPU devices to use'
# )
parser.add_argument("--gpu", type=int, default=0, help="GPU index to use (default: 0)")
parser.add_argument(
    "--in_dim",
    type=int,
    default=12,
    help="Input dimension from sample-level VAF dataset (default: 12)",
)
parser.add_argument(
    "--out_dim", type=int, default=4, help="Model output dimension (default: 4)"
)
parser.add_argument(
    "--attn_dim",
    type=int,
    default=512,
    help="Model dimension after numerical_embedder and encoder (default: 512)",
)
parser.add_argument(
    "--depth", type=int, default=4, help="Number of layers to use (default: 4)"
)
parser.add_argument(
    "--heads", type=int, default=16, help="Number of attention heads (Default: 16)"
)
parser.add_argument(
    "--coef",
    type=float,
    default=1e-5,
    help="Regularization for loss function (Default: 1e-5)",
)
parser.add_argument(
    "--max_epoch",
    type=int,
    default=100,
    help="Maximum epochs for training (default: 100)",
)
parser.add_argument(
    "--patience",
    type=int,
    default=25,
    help="Patience # for model evaluation epochs (default: 25)",
)
parser.add_argument(
    "--nfolds", type=int, default=7, help="Number of training folds (default: 7)"
)

args = parser.parse_args()

settings = {
    "in_dim": args.in_dim,
    "out_dim": args.out_dim,
    "attn_dim": args.attn_dim,
    "n_attn_heads": args.heads,
    "depth": args.depth,
    "coef": args.coef,
    "num_folds": args.nfolds,
    "patience": args.patience,
    "max_epoch": args.max_epoch,
    "gpu": args.gpu,
}


def main(args, i):
    # log_dir = f"{args.output_dir}/AD{args.attn_dim}_H{args.heads}_L{args.depth}_coef{args.coef}/logs"
    # ckpt_dir = f"{args.output_dir}/AD{args.attn_dim}_H{args.heads}_L{args.depth}_coef{args.coef}/checkpoints/{i+1}"

    log_dir = f"{args.output_dir}/noPE_AD{args.attn_dim}_H{args.heads}_L{args.depth}_coef{args.coef}/logs"
    ckpt_dir = f"{args.output_dir}/noPE_AD{args.attn_dim}_H{args.heads}_L{args.depth}_coef{args.coef}/checkpoints/{i+1}"

    dm = TabularDataModule(
        root_file=args.data_dir,
        time_bins_file=args.time_bins_fp,
        shuffle_dataset=True,
        num_workers=4,
        pin_memory=False,
        batch_size=1,
    )

    model = EngineAttn(
        input_dim=args.in_dim,
        out_dim=args.out_dim,
        attn_dim=args.attn_dim,
        n_heads=args.heads,
        depth=args.depth,
        coef=args.coef,
        fold_num=i,
    )

    tb_logger = TensorBoardLogger(save_dir=log_dir)

    trainer = pl.Trainer(
        max_epochs=args.max_epoch,
        accelerator="gpu",
        devices=[args.gpu],
        logger=tb_logger,
        callbacks=[
            ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="{epoch}--{avg_val_loss:.4f}",
                save_weights_only=True,
                mode="min",
                monitor="checkpoint_on",
            ),
            LearningRateMonitor("epoch"),
            EarlyStopping(
                monitor="checkpoint_on",
                min_delta=0.01,
                patience=args.patience,
                mode="min",
            ),
        ],
    )

    start = time.time()
    trainer.fit(model, dm)
    print(f"Time elapsed: {time.time() - start}")

    trainer.test(model, dm, ckpt_path="best")
    results = model.test_results
    return results


if __name__ == "__main__":
    torch.set_float32_matmul_precision(precision="high")
    # torch.autograd.set_detect_anomaly(True)
    # results_path = f"{args.output_dir}/AD{args.attn_dim}_H{args.heads}_L{args.depth}_coef{args.coef}/fold_results.pkl"

    results_path = f"{args.output_dir}/noPE_AD{args.attn_dim}_H{args.heads}_L{args.depth}_coef{args.coef}/fold_results.pkl"

    for k, v in settings.items():
        print(f"{k}: {v}")

    num_kfolds = args.nfolds

    all_results = []
    for i in range(num_kfolds):
        test_results = main(args, i)
        all_results.append(test_results)

    all_results_dict = dict()
    for i in range(num_kfolds):
        fold_num = i + 1
        all_results_dict[f"fold_{fold_num}"] = all_results[i]

    pickle.dump(all_results_dict, open(results_path, "wb"))
