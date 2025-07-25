import os
from datetime import datetime
import matplotlib.pyplot as plt

def init_experiment_log_dir(base_dir: str = "experiments") -> str:
    """
    Create a directory named with the date and the hour of the experiment

    :param base_dir: the path in which you would like to save your experiments data
    :return: the absolute path of the directory
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_path = os.path.join(base_dir, timestamp)

    os.makedirs(exp_path, exist_ok=True)
    return exp_path


def save_plot(fig, save_dir: str, filename: str = "plot.png", dpi: int = 300):
    """
    Sauvegarde un plot matplotlib dans le dossier donné.

    :param fig: matplotlib.figure.Figure object.
    :param save_dir: save directory (doit exister).
    :param filename: filename (ex: "loss.png").
    :param dpi: Resolution.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"[✔] Plot sauvegardé → {save_path}")


import torch

def save_model_weights(model, save_dir: str, filename: str = "model.pth"):
    """
    Sauvegarde les poids d’un modèle PyTorch dans le dossier spécifié.

    :param model: The PyTorch model (nn.Module).
    :param save_dir: Directory in which you would like to save the model (must exist).
    :param filename: name of the file.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), save_path)
    print(f"[✔] Modèle sauvegardé → {save_path}")


def save_args(args: dict, save_dir: str, filename: str = "args.txt"):
    """
    Sauvegarde un dictionnaire de paramètres dans un fichier texte.

    :param args: Dictionnaire des arguments/paramètres.
    :param save_dir: Dossier de sauvegarde (sera créé si nécessaire).
    :param filename: Nom du fichier texte (par défaut: "args.txt").
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    with open(save_path, 'w') as f:
        for key, value in args.items():
            f.write(f"{key}: {value}\n")

    print(f"[✔] Paramètres sauvegardés → {save_path}")