import yaml
import argparse
import os
import datetime
import torch
import torch.optim
from torch.utils.data.dataloader import DataLoader

# Local imports
from dataloaders.clotho_dataloader import AudioCaptioningDataset, load_data_from_npy
from models.clip import BaseClip, ViTClip, PANNClip
from util.utils import eval_model_embeddings
import wandb

parser = argparse.ArgumentParser(description="Music caption retrieval project")
parser.add_argument("--config", default="configs/resnet.yaml")
parser.add_argument("--mode", default="train", choices=["train", "test"])

def set_syspath(sys_path: str, model_lib_path: str):
    """
    If needed, you can dynamically modify sys.path here. 
    For now, we omit usage if your project is structured as a proper package.
    """
    pass

def train_model(
    args,
    get_metrics: bool = False,
    eval_batch_size: int = 512,
    print_every_epoch: int = 1
):
    """
    Train the CLIP-like model on audio-caption data.

    Args:
        get_metrics (bool): If True, evaluate MRR, MAP@K, and R@K after each epoch.
        eval_batch_size (int): Batch size for metric evaluation.
        print_every_epoch (int): Print metrics every n epochs.
    """
    config = {"lr": args.lr, "batch_size": args.batch_size, "seed": args.random_seed}
    wandb.init(project=args.model + "-F22", entity="deep-learning-f22", config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device=device)

    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)

    print(f"Running on device: {device}")

    model_dir = args.save_dir
    if args.random_seed is not None:
        model_dir += f"seed_{args.random_seed}_{datetime.datetime.now():%Y%m%d%H%M%S}"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=1, factor=0.8
    )

    print("Creating dataloaders...")
    data_train = load_data_from_npy(args.data_dir, 'train/val')

    if args.model == 'PANN':
        train_dataset = AudioCaptioningDataset(
            data_train['train_spectrograms'], data_train['train_captions'], augment=True
        )
        val_dataset = AudioCaptioningDataset(
            data_train['val_spectrograms'], data_train['val_captions']
        )
    else:
        train_dataset = AudioCaptioningDataset(
            data_train['train_spectrograms'], data_train['train_captions'],
            augment=True, multichannel=True
        )
        val_dataset = AudioCaptioningDataset(
            data_train['val_spectrograms'], data_train['val_captions'], multichannel=True
        )

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    train_dataloader_metrics = DataLoader(train_dataset, batch_size=eval_batch_size, shuffle=False)
    val_dataloader_metrics = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)

    min_val_loss = float('inf')

    print("Starting training!")
    for e in range(args.epochs):
        start = datetime.datetime.now()
        print(f"Beginning epoch {e+1} at {start}")
        model.train()

        train_total_loss = 0.0
        for idx, batch in enumerate(train_dataloader):
            if idx % 10 == 0:
                print(f"Training batch {idx+1} at {datetime.datetime.now()}")

            batch = (batch[0].to(device), batch[1].to(device), batch[2].to(device))
            batch_loss, _, _ = model(batch)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            lr_scheduler.step(batch_loss)

            train_total_loss += batch_loss.item()

        avg_train_loss = train_total_loss / len(train_dataloader)
        print(f"Training Loss: {avg_train_loss:.4f}")
        wandb.log({'Training Loss': avg_train_loss})

        # Evaluate on training set
        if get_metrics:
            train_metrics = evaluate(model, train_dataloader_metrics, "train")
            wandb.log({
                'train MRR': train_metrics["train_MRR"],
                'train MAP@K': train_metrics["train_MAP@K"],
                'train R@K': train_metrics["train_R@K"]
            })
            if (e + 1) % print_every_epoch == 0 or (e + 1) == args.epochs:
                print(
                    "Train Metrics [Epoch {}/{}]: MRR={:.4f}, MAP@K={:.4f}, R@K={:.4f}".format(
                        e+1, args.epochs,
                        train_metrics["train_MRR"],
                        train_metrics["train_MAP@K"],
                        train_metrics["train_R@K"]
                    )
                )
            with open(os.path.join(model_dir, "train_metrics.txt"), "w") as f:
                for k, v in train_metrics.items():
                    f.write(f"{k}: {v}\n")

        # Validation step
        model.eval()
        val_total_loss = 0.0
        with torch.no_grad():
            for idx, batch in enumerate(val_dataloader):
                if idx % 10 == 0:
                    print(f"Validation batch {idx+1} at {datetime.datetime.now()}")
                batch = (batch[0].to(device), batch[1].to(device), batch[2].to(device))
                batch_loss, _, _ = model(batch)
                val_total_loss += batch_loss.item()

        avg_val_loss = val_total_loss / len(val_dataloader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        wandb.log({'Validation Loss': avg_val_loss})

        if get_metrics:
            val_metrics = evaluate(model, val_dataloader_metrics, "val")
            wandb.log({
                'val MRR': val_metrics["val_MRR"],
                'val MAP@K': val_metrics["val_MAP@K"],
                'val R@K': val_metrics["val_R@K"]
            })
            if (e + 1) % print_every_epoch == 0 or (e + 1) == args.epochs:
                print(
                    "Val Metrics [Epoch {}/{}]: MRR={:.4f}, MAP@K={:.4f}, R@K={:.4f}".format(
                        e+1, args.epochs,
                        val_metrics["val_MRR"],
                        val_metrics["val_MAP@K"],
                        val_metrics["val_R@K"]
                    )
                )
            with open(os.path.join(model_dir, "val_metrics.txt"), "w") as f:
                for k, v in val_metrics.items():
                    f.write(f"{k}: {v}\n")

        # Save best model
        if avg_val_loss < min_val_loss:
            print("New best model found. Saving...")
            save_filename = os.path.join(model_dir, f"model_{e+1}.pth")
            torch.save(model.state_dict(), save_filename)
            print(f"Saved model checkpoint to {save_filename}")
            min_val_loss = avg_val_loss


def evaluate(model: torch.nn.Module, data_loader: DataLoader, stage: str = 'train'):
    """
    Evaluate a model on a given DataLoader and return the 
    standard metrics (MRR, MAP@K, R@K).

    Args:
        model (nn.Module): The model to evaluate.
        data_loader (DataLoader): The data loader to evaluate on.
        stage (str): 'train', 'val', or 'test'.
    
    Returns:
        dict: {f"{stage}_MRR", f"{stage}_MAP@K", f"{stage}_R@K"}
    """
    device = next(model.parameters()).device
    metrics = eval_model_embeddings(model, device, data_loader, ["MRR", "MAP@K", "R@K"], k=10)

    # rename metrics keys
    stage_metrics = {}
    for k, v in metrics.items():
        stage_metrics[f"{stage}_{k}"] = v
    return stage_metrics


def load_model(model_type: str, device: torch.device, state_dict: str = None) -> torch.nn.Module:
    """
    Instantiate and optionally load a model checkpoint.

    Args:
        model_type (str): 'ResNet', 'ViT', or 'PANN'.
        device (torch.device): CPU or GPU device.
        state_dict (str): Path to a checkpoint file, if any.

    Returns:
        nn.Module: The instantiated (and optionally loaded) model.
    """
    if model_type == "ResNet":
        model = BaseClip(device=device, fine_tune=args.fine_tune)
    elif model_type == "ViT":
        model = ViTClip(device=device, fine_tune=args.fine_tune)
    elif model_type == "PANN":
        model = PANNClip(device=device, fine_tune=args.fine_tune)
    else:
        raise ValueError("Unknown model type. Choose from [ResNet, ViT, PANN].")

    if state_dict is not None:
        loaded_dict = torch.load(state_dict, map_location=device)
        model.load_state_dict(loaded_dict)

    return model


def main():
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    # If you need to modify sys.path for custom imports:
    # set_syspath(args.sys_path, args.model_lib_path)

    if args.mode == "train":
        train_model(get_metrics=args.get_metrics, eval_batch_size=512, print_every_epoch=1)
    elif args.mode == "test":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(args.model, device, args.checkpoint_path)

        data_test = load_data_from_npy(args.data_dir, 'test')
        if args.model == 'PANN':
            test_dataset = AudioCaptioningDataset(
                data_test['test_spectrograms'], data_test['test_captions']
            )
        else:
            test_dataset = AudioCaptioningDataset(
                data_test['test_spectrograms'], data_test['test_captions'],
                multichannel=True
            )

        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        test_metrics = evaluate(model, test_dataloader, stage="test")

        save_path = os.path.join(args.save_dir, f"{args.model}_test_metrics.txt")
        print(
            "Test Metrics: MRR={:.4f}, MAP@K={:.4f}, R@K={:.4f}".format(
                test_metrics["test_MRR"],
                test_metrics["test_MAP@K"],
                test_metrics["test_R@K"]
            )
        )
        with open(save_path, "w") as f:
            for k, v in test_metrics.items():
                f.write(f"{k}: {v}\n")


if __name__ == "__main__":
    main()
