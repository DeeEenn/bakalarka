import torch

from asformer_model import ASFormer
from ms_tcn_model import MSTCN


MODEL_CONFIGS = {
    "asformer": {
        "checkpoint": "asformer_attention_v1.pth",
        "kwargs": {
            "num_layers": 8,
            "d_model": 128,
            "input_dim": 243,
            "num_classes": 6,
            "num_heads": 8,
            "dropout": 0.1,
            "max_dilation": 16,
        },
    },
    "mstcn": {
        "checkpoint": "mstcn_v1.pth",
        "kwargs": {
            "num_stages": 4,
            "num_layers": 8,
            "num_f_maps": 64,
            "dim_in": 243,
            "num_classes": 6,
            "dropout": 0.3,
        },
    },
}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(model_name: str):
    model_name = model_name.lower()
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_name: {model_name}. Use one of: {list(MODEL_CONFIGS.keys())}")

    cfg = MODEL_CONFIGS[model_name]["kwargs"]

    if model_name == "asformer":
        model = ASFormer(**cfg)
    elif model_name == "mstcn":
        model = MSTCN(**cfg)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    return model


def load_model(model_name: str, checkpoint_path: str = None, device=None):
    if device is None:
        device = get_device()

    model_name = model_name.lower()
    model = build_model(model_name).to(device)

    if checkpoint_path is None:
        checkpoint_path = MODEL_CONFIGS[model_name]["checkpoint"]

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint_path