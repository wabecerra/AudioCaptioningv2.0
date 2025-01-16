import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights
from transformers import RobertaTokenizer

from . import text_encoder
from . import audio_encoders
from ..util.loss import InfoNCE


class ProjectionHead(nn.Module):
    """
    A two-layer projection head with residual + layer norm, typically
    used to map embeddings to a common dimension in CLIP-like models.
    """
    def __init__(
        self,
        embedding_dim: int,
        projection_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim).double()
        self.relu = nn.ReLU()
        self.fc = nn.Linear(projection_dim, projection_dim).double()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim).double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, x.shape[-1]).double()
        projected = self.projection(x)
        x = self.relu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected  # Residual connection
        x = self.layer_norm(x)
        return x


class ViTClip(nn.Module):
    """
    A CLIP-like model using a Vision Transformer as the audio encoder 
    (treating spectrograms like images) and RoBERTa as the text encoder.
    """
    def __init__(
        self,
        device: torch.device,
        image_embedding_size: int = 768,
        text_embedding_size: int = 768,
        audio_encoder: nn.Module = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1),
        fine_tune: bool = False
    ):
        super().__init__()
        self.device = device

        self.audio_encoder = audio_encoder.to(device)
        self.text_encoder = text_encoder.TextEncoder().to(device)
        self.audio_projection = ProjectionHead(embedding_dim=image_embedding_size).to(device)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding_size).to(device)

        self.transforms = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()
        self.fine_tune = fine_tune
        self.audio_size = image_embedding_size
        self.layer_extract = [
            "heads.head",
            "encoder.layers.encoder_layer_11.mlp.4",
            "encoder.ln"
        ]

        # Save output of forward hooks
        self._features = {layer: torch.empty(0) for layer in self.layer_extract}
        self.register_hooks()

        if fine_tune:
            # Only allow gradient on certain layers
            params_to_fine_tune = [
                "audio_encoder.heads", 
                "audio_encoder.encoder.layers.encoder_layer_11",
                "audio_encoder.encoder.layers.encoder_layer_10",
                "audio_projection", 
                "text_projection"
            ]
            for name, param in self.named_parameters():
                if any(p in name for p in params_to_fine_tune):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def register_hooks(self) -> None:
        """
        Registers forward hooks on the specified layers to capture their outputs.
        """
        for layer_id in self.layer_extract:
            layer_module = dict(self.audio_encoder.named_modules())[layer_id]
            layer_module.register_forward_hook(self.save_output_hook(layer_id))

    def save_output_hook(self, layer_id: str):
        """
        A forward hook function that captures the output of a given layer
        and stores it in the self._features dict.
        """
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, batch: tuple):
        """
        Forward pass for a batch of (audio_spectrogram, input_ids, attention_mask).

        Args:
            batch: A tuple of (spectrogram, input_ids, attention_mask).

        Returns:
            A tuple of (loss, audio_embeddings, text_embeddings).
        """
        audio_specs, input_ids, attention_mask = batch
        batch_size = audio_specs.shape[0]

        # Text embeddings
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        with torch.no_grad() if not self.fine_tune else torch.enable_grad():
            text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Audio embeddings
        processed_audio = torch.zeros((batch_size, self.audio_size), device=self.device)
        if not self.fine_tune:
            with torch.no_grad():
                for i in range(batch_size):
                    transformed_audio = self.transforms(audio_specs[i, :, :, :].squeeze(0))
                    features = audio_encoders.get_vit_feature_vector(self.audio_encoder, self.device, transformed_audio.type(torch.DoubleTensor))
                    processed_audio[i, :] = features[0, :]
        else:
            for i in range(batch_size):
                transformed_audio = self.transforms(audio_specs[i, :, :, :].unsqueeze(0))
                _ = self.audio_encoder(transformed_audio)
                # The last hook layer's output
                processed_audio[i, :] = self._features["encoder.layers.encoder_layer_11.mlp.4"][0, :]

        audio_embeddings = self.audio_projection(processed_audio.float())
        text_embeddings = self.text_projection(text_features)

        # Contrastive loss
        loss_fn = InfoNCE()
        batch_loss = loss_fn(text_embeddings, audio_embeddings)
        return batch_loss, audio_embeddings, text_embeddings


class BaseClip(nn.Module):
    """
    A CLIP-like model using ResNet-50 as the audio encoder 
    (treating spectrograms as images) and RoBERTa as text encoder.
    """
    def __init__(
        self,
        device: torch.device,
        image_embedding_size: int = 1000,
        text_embedding_size: int = 768,
        audio_encoder: nn.Module = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2),
        fine_tune: bool = False
    ):
        super().__init__()
        self.device = device
        self.fine_tune = fine_tune
        self.audio_encoder = audio_encoder.to(device)
        self.text_encoder = text_encoder.TextEncoder().to(device)

        self.audio_projection = ProjectionHead(embedding_dim=image_embedding_size).to(device)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding_size).to(device)
        self.audio_size = image_embedding_size

        # capture output of last fc layer
        self.layer_extract = ["fc"]
        self._features = {layer: torch.empty(0) for layer in self.layer_extract}
        self.register_hooks()

        if fine_tune:
            # Only allow gradient on certain layers
            params_to_fine_tune = [
                "audio_encoder.layer4.0",
                "audio_encoder.layer4.1",
                "audio_encoder.layer4.2",
                "audio_encoder.fc"
            ]
            for name, param in self.named_parameters():
                if any(p in name for p in params_to_fine_tune):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def register_hooks(self) -> None:
        for layer_id in self.layer_extract:
            layer_module = dict(self.audio_encoder.named_modules())[layer_id]
            layer_module.register_forward_hook(self.save_output_hook(layer_id))

    def save_output_hook(self, layer_id: str):
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, batch: tuple):
        """
        Forward pass for a batch of (audio_spectrogram, input_ids, attention_mask).
        """
        audio_specs, input_ids, attention_mask = batch

        # reshape text inputs
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        batch_size = audio_specs.shape[0]

        text_features = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        processed_audio = torch.zeros((batch_size, self.audio_size), device=self.device)
        if not self.fine_tune:
            with torch.no_grad():
                for i in range(batch_size):
                    _ = self.audio_encoder(audio_specs[i, :, :, :].unsqueeze(0))
                    processed_audio[i, :] = self._features["fc"]
        else:
            for i in range(batch_size):
                _ = self.audio_encoder(audio_specs[i, :, :, :].unsqueeze(0))
                processed_audio[i, :] = self._features["fc"]

        audio_embeddings = self.audio_projection(processed_audio)
        text_embeddings = self.text_projection(text_features)

        loss_fn = InfoNCE()
        batch_loss = loss_fn(text_embeddings, audio_embeddings)
        return batch_loss, audio_embeddings, text_embeddings


class PANNClip(nn.Module):
    """
    A CLIP-like model using a PANN CNN14 backbone as the audio encoder and RoBERTa for text.
    """
    def __init__(
        self,
        device: torch.device,
        temp: float = 1.0,
        image_embedding_size: int = 2048,
        text_embedding_size: int = 768,
        audio_encoder: nn.Module = audio_encoders.Cnn14(),
        model_path: str = "/content/drive/MyDrive/MusicCaptioning/Nikhil/models/pretrained_weights/Cnn14_mAP=0.431.pth",
        fine_tune: bool = False
    ):
        super().__init__()
        self.device = device
        self.fine_tune = fine_tune
        self.temperature = temp
        self.text_encoder = text_encoder.TextEncoder().to(device)

        self.audio_encoder = audio_encoder.to(device)
        self.audio_projection = ProjectionHead(embedding_dim=image_embedding_size).to(device)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding_size).to(device)

        self.saved_model = torch.load(model_path)
        # Remove keys not used
        self.saved_model['model'].pop('spectrogram_extractor.stft.conv_real.weight', None)
        self.saved_model['model'].pop('spectrogram_extractor.stft.conv_imag.weight', None)
        self.saved_model['model'].pop('logmel_extractor.melW', None)
        self.saved_model['model'].pop('fc1.weight', None)
        self.saved_model['model'].pop('fc1.bias', None)
        self.saved_model['model'].pop('fc_audioset.weight', None)
        self.saved_model['model'].pop('fc_audioset.bias', None)
        self.audio_encoder.load_state_dict(self.saved_model["model"])

    def forward(self, batch: tuple):
        audio_specs, input_ids, attention_mask = batch
        # PANN audio encoder
        audio_features = self.audio_encoder(audio_specs)

        # text features
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        with torch.no_grad() if not self.fine_tune else torch.enable_grad():
            text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

        audio_embeddings = self.audio_projection(audio_features)
        text_embeddings = self.text_projection(text_features)

        loss_fn = InfoNCE(temp=self.temperature)
        batch_loss = loss_fn(text_embeddings, audio_embeddings)
        return batch_loss, audio_embeddings, text_embeddings
