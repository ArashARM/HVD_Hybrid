import os
import math
import time
import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyvista as pv
import torch
import torch.nn.functional as F

from tqdm import trange, tqdm
from matplotlib.patches import Rectangle
from torch.utils.tensorboard import SummaryWriter
from scipy.spatial import (
    Delaunay,
    Voronoi,
    voronoi_plot_2d,
    cKDTree,
    QhullError,
)

from Utils.CADTensorGenerator import CADTensorGenerator
from Utils.CADDomainVisualizer import CADDomainVisualizer
from Utils.CADVisualizer import CADVisualizer
from neuraltomo_fem import run_fem_loss
from problems.ThickenShell import ThickenShell
from Training.MainTrain import NN_Trainer, TrainingConfig
from Decoder_CLasses.ContinuousVoronoiDecoder import (
    ContinuousVoronoiDecoder,
)
from Decoder_CLasses.NearUniformHoneycombBaseline import (
    NearUniformHoneycombBaseline,
)
from Decoder_CLasses.IntrinsicSurfaceHoneycombBaseline import (
    IntrinsicSurfaceHoneycombBaseline,
)
from Pred_NN_Classes.ppnet import PPNet
from Utils.ExportAbaqus_VoxelBased import export_abaqus_voxel_fem
from Utils.ExportAbaqus import export_abaqus_shell


def random_seeds_min_dist(
    number,
    min_dist=0.08,
    seed=1,
    max_tries=10000,
    device="cpu",
):
    torch.manual_seed(seed)

    seeds = []
    tries = 0

    while len(seeds) < number and tries < max_tries:
        point = torch.rand(2, device=device)

        if not seeds:
            seeds.append(point)
        else:
            current = torch.stack(seeds, dim=0)
            distances = torch.linalg.norm(
                current - point[None, :],
                dim=-1,
            )

            if distances.min() >= min_dist:
                seeds.append(point)

        tries += 1

    if len(seeds) < number:
        raise RuntimeError(
            f"Could only generate {len(seeds)} seeds "
            f"with min_dist={min_dist}."
        )

    return torch.stack(seeds, dim=0)


def main():
    # ---------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------
    seed = 20

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # ---------------------------------------------------------
    # Paths
    # ---------------------------------------------------------
    base_directory = Path(__file__).resolve().parent
    test_parts_directory = base_directory / "Testparts"

    print("Code directory:", base_directory)
    print("Test STEP files directory:", test_parts_directory)

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # For a standalone script, do not use a Jupyter backend.
    pv.OFF_SCREEN = False

    # ---------------------------------------------------------
    # Continue your CAD/FEM/training code here
    # ---------------------------------------------------------

    # Example:
    # shape_path = test_parts_directory / "FreeSharp2.stp"
    #
    # ...
    #
    # result = trainer.train(
    #     shape_path=shape_path,
    #     face_tensors=[face_mesh],
    # )


if __name__ == "__main__":
    main()