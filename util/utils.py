import torch
import torch.nn.functional as F
from typing import List, Dict

def load_pretrained_img_model(model_class, device: torch.device, checkpoint_path: str):
    """
    Load a pretrained model from a specified checkpoint.

    Args:
        model_class: A callable that returns an uninitialized model instance.
        device (torch.device): CPU or GPU device.
        checkpoint_path (str): Path to the saved checkpoint file.

    Returns:
        A model instance loaded with weights from the checkpoint.
    """
    model = model_class()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    return model


def eval_model_embeddings(
    model: torch.nn.Module,
    device: torch.device,
    data_loader: torch.utils.data.DataLoader,
    metric_names: List[str],
    **kwargs
) -> Dict[str, float]:
    """
    Evaluate the model on given metrics. 
    Supported metrics are: ['MRR', 'MAP@K', 'R@K'].

    Args:
        model (nn.Module): The model to evaluate.
        device (torch.device): The device to run the model on.
        data_loader (DataLoader): The data loader providing evaluation data.
        metric_names (List[str]): List of metric names to compute.
        kwargs: Additional arguments for certain metrics (e.g. 'k=10').

    Returns:
        dict: e.g. {"MRR": 0.5, "MAP@K": 0.3, "R@K": 0.4}
    """
    model.to(device)
    model.eval()

    accumulated_metrics = {"MRR": [], "MAP@K": [], "R@K": []}

    for idx, batch in enumerate(data_loader):
        batch = (batch[0].to(device), batch[1].to(device), batch[2].to(device))
        with torch.no_grad():
            _, audio_emb, text_emb = model.forward(batch)

        if 'MRR' in metric_names:
            accumulated_metrics["MRR"].append(mean_reciprocal_rank(audio_emb, text_emb))
        if 'MAP@K' in metric_names:
            if 'k' not in kwargs:
                raise ValueError("MAP@K requires 'k' in kwargs.")
            k_val = kwargs['k']
            accumulated_metrics["MAP@K"].append(mean_avg_precision_at_k(audio_emb, text_emb, k=k_val))
        if 'R@K' in metric_names:
            if 'k' not in kwargs:
                raise ValueError("R@K requires 'k' in kwargs.")
            k_val = kwargs['k']
            accumulated_metrics["R@K"].append(mean_recall_at_k(audio_emb, text_emb, k=k_val))

    # Average across batches
    final_metrics = {}
    for m in metric_names:
        # Avoid empty lists if metric wasn't used
        if accumulated_metrics[m]:
            final_metrics[m] = sum(accumulated_metrics[m]) / len(accumulated_metrics[m])
    return final_metrics


def mean_reciprocal_rank(
    audio_embeddings: torch.Tensor,
    caption_embeddings: torch.Tensor
) -> float:
    """
    Compute mean reciprocal rank for retrieval tasks:
    For each audio embedding, rank all caption embeddings and see 
    where the matching one appears.

    NOTE: This function depends on how you define "match". 
    Currently, it assumes repeated embeddings for positive pairs.

    Returns:
        float: mean reciprocal rank across the batch.
    """
    audio_norm = F.normalize(audio_embeddings, p=2, dim=-1)
    text_norm = F.normalize(caption_embeddings, p=2, dim=-1)
    sim_matrix = audio_norm @ text_norm.T

    # Identify unique audio embeddings
    unique_audio = torch.unique(audio_norm, dim=0)
    unique_indices = []
    for i in range(unique_audio.size(0)):
        indices = torch.where(torch.all(audio_norm == unique_audio[i], dim=1))[0]
        unique_indices.append(indices)

    mask = torch.zeros_like(sim_matrix)
    for ui in unique_indices:
        for j in range(ui.size(0)):
            for h in range(j, ui.size(0)):
                mask[ui[j], ui[h]] = 1
                mask[ui[h], ui[j]] = 1

    # Sort each row in descending order
    _, sorted_indices = torch.sort(sim_matrix, dim=1, descending=True)
    # Then figure out the highest rank index that has mask=1
    # The rank is (index + 1)
    ranks = []
    for i in range(sorted_indices.size(0)):
        row_mask = mask[i][sorted_indices[i]]
        # find the first position where row_mask=1
        pos = (row_mask == 1).nonzero(as_tuple=True)[0]
        if len(pos) > 0:
            rank = pos[0].item() + 1
            ranks.append(1.0 / rank)
        else:
            ranks.append(0.0)

    return sum(ranks) / len(ranks)


def mean_avg_precision_at_k(
    audio_embeddings: torch.Tensor,
    caption_embeddings: torch.Tensor,
    k: int = 10
) -> float:
    """
    Compute mean average precision at K for retrieval tasks.

    Returns:
        float: MAP@K
    """
    audio_norm = F.normalize(audio_embeddings, p=2, dim=-1)
    text_norm = F.normalize(caption_embeddings, p=2, dim=-1)
    sim_matrix = audio_norm @ text_norm.T

    unique_audio = torch.unique(audio_norm, dim=0)
    unique_indices = []
    for i in range(unique_audio.size(0)):
        indices = torch.where(torch.all(audio_norm == unique_audio[i], dim=1))[0]
        unique_indices.append(indices)

    mask = torch.zeros_like(sim_matrix)
    for ui in unique_indices:
        for j in range(ui.size(0)):
            for h in range(j, ui.size(0)):
                mask[ui[j], ui[h]] = 1
                mask[ui[h], ui[j]] = 1

    _, sorted_indices = torch.sort(sim_matrix, dim=1, descending=True)
    # Reorder mask rows
    row_indices = torch.arange(mask.size(0)).unsqueeze(1)
    mask_sorted = mask[row_indices, sorted_indices]
    # Precision@K = relevant@K / K
    # Then average across all queries
    avg_precision = (mask_sorted[:, :k].sum(dim=1) / k).mean().item()
    return avg_precision


def mean_recall_at_k(
    audio_embeddings: torch.Tensor,
    caption_embeddings: torch.Tensor,
    k: int = 10
) -> float:
    """
    Compute mean recall at K for retrieval tasks.

    Returns:
        float: recall@K
    """
    audio_norm = F.normalize(audio_embeddings, p=2, dim=-1)
    text_norm = F.normalize(caption_embeddings, p=2, dim=-1)
    sim_matrix = audio_norm @ text_norm.T

    unique_audio = torch.unique(audio_norm, dim=0)
    unique_indices = []
    for i in range(unique_audio.size(0)):
        indices = torch.where(torch.all(audio_norm == unique_audio[i], dim=1))[0]
        unique_indices.append(indices)

    mask = torch.zeros_like(sim_matrix)
    for ui in unique_indices:
        for j in range(ui.size(0)):
            for h in range(j, ui.size(0)):
                mask[ui[j], ui[h]] = 1
                mask[ui[h], ui[j]] = 1

    _, sorted_indices = torch.sort(sim_matrix, dim=1, descending=True)
    row_indices = torch.arange(mask.size(0)).unsqueeze(1)
    mask_sorted = mask[row_indices, sorted_indices]

    # Recall@K = relevant@K / total_relevant
    # total_relevant = sum of mask in that row
    total_relevant = mask.sum(dim=1)  # for each row
    recall_values = []
    for i in range(mask_sorted.size(0)):
        retrieved = mask_sorted[i, :k].sum()
        recall = retrieved / (total_relevant[i] + 1e-8)  # avoid division by zero
        recall_values.append(recall)
    return float(torch.mean(torch.tensor(recall_values)))
