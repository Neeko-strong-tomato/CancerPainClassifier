# CancerPainClassifier
# Copyright (c) 2025 Neeko
# License: MIT
# If used in research, please cite: https://github.com/Neeko-strong-tomato/CancerPainClassifier

import logging
import sys
from colorlog import ColoredFormatter
from collections import Counter

def setup_logger(name: str = "app", level: int = logging.INFO) -> logging.Logger:
    """
    Configure un logger coloré pour affichage dans le terminal.

    :param name: Nom du logger.
    :param level: Niveau de log à afficher.
    :return: logger object, ready to print
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)

        formatter = ColoredFormatter(
            fmt="%(log_color)s%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG":    "cyan",
                "INFO":     "green",
                "WARNING":  "yellow",
                "ERROR":    "red",
                "CRITICAL": "bold_red",
            }
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


class ProgressReporter:
    def __init__(self, bar_name, step=3.5):
        self.last_progress = -100.0
        self.step = step
        self.name = bar_name             

    def update(self, percent):
        if percent - self.last_progress >= self.step:
            self.last_progress = percent
            filled = int(percent / 2)
            bar = '#' * filled + '-' * (50 - filled)
            print(f"\r {self.name} : [{bar}] {percent:5.1f}%", end='', flush=True)

        elif percent == 100 :
            filled = int(percent / 2)
            bar = '#' * filled + '-' * (50 - filled)
            print(f"\r {self.name} : [{bar}] {percent:5.1f}%", end='', flush=True)


def log_class_distribution(Y, title="Class Distribution"):
    """
    Affiche un tableau récapitulatif de la distribution des classes dans Y.
    
    Args:
        Y: list ou array de labels (int)
        title: str, titre du tableau
    """
    counts = Counter(Y)
    total = sum(counts.values())

    # En-tête
    print("\n" + "=" * 50)
    print(f"{title:^50}")
    print("=" * 50)
    print(f"| {'Class':^10} | {'Patient Count':^15} | {'Proportion (%)':^15} |")
    print("-" * 50)

    for cls in sorted(counts.keys()):
        count = counts[cls]
        proportion = 100 * count / total
        print(f"| {str(cls):^10} | {str(count):^15} | {proportion:^15.2f} |")

    print("=" * 50 + "\n")
