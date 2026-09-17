#!/usr/bin/env python
# coding: utf-8

# In[1]:



import os
import math
import time
import sys
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import torch
import torch.nn.functional as F  
from torch.utils.tensorboard import SummaryWriter

from Utils.CADTensorGenerator import CADTensorGenerator
from Utils.CADDomainVisualizer import CADDomainVisualizer
from Utils.CADVisualizer   import CADVisualizer
from neuraltomo_fem import run_fem_loss
from problems.ThickenShell import ThickenShell
from Training.MainTrain import NN_Trainer, TrainingConfig
from Decoder_CLasses.ContinuousVoronoiDecoder import ContinuousVoronoiDecoder
from Decoder_CLasses.NearUniformHoneycombBaseline import NearUniformHoneycombBaseline
from Decoder_CLasses.IntrinsicSurfaceHoneycombBaseline import IntrinsicSurfaceHoneycombBaseline
from Pred_NN_Classes.ppnet import PPNet
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d, cKDTree,QhullError
from Utils.ExportAbaqus_VoxelBased import export_abaqus_voxel_fem
from Utils.ExportAbaqus import export_abaqus_shell

import pyvista as pv


# ---- Reproducibility (recommended for D_params comparisons) ----
SEED = 20
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
print("Code Directory:", BASE)
TesPartsDir = BASE / "Testparts" 
print("Test Step files Directory:", TesPartsDir)


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("device:", device)
# -------- PYVISTA BACKEND --------
def setup_pyvista(device):
    is_mac = sys.platform == "darwin"

    # Mac + MPS: prefer static to avoid VTK/trame hangs
    if is_mac :
        pv.OFF_SCREEN = True
        pv.set_jupyter_backend("static")
        backend = "static"
    else:
        try:
            pv.set_jupyter_backend("trame")
            backend = "trame"
        except Exception:
            pv.OFF_SCREEN = True
            pv.set_jupyter_backend("static")
            backend = "static"

    print(f"PyVista backend: {backend}")

setup_pyvista(device)


def random_seeds_min_dist(N, min_dist=0.08, seed=1, max_tries=10000, device="cpu"):
    torch.manual_seed(seed)

    seeds = []
    tries = 0

    while len(seeds) < N and tries < max_tries:
        p = torch.rand(2, device=device)

        if len(seeds) == 0:
            seeds.append(p)
        else:
            current = torch.stack(seeds, dim=0)
            d = torch.linalg.norm(current - p[None, :], dim=-1)

            if d.min() >= min_dist:
                seeds.append(p)

        tries += 1

    if len(seeds) < N:
        raise RuntimeError(
            f"Could only generate {len(seeds)} seeds with min_dist={min_dist}."
        )

    return torch.stack(seeds, dim=0)

# In[2]:


# Laoding model and extracting mesh and tensors as input
FreeFormSurf1  = TesPartsDir / "FreeFormCrv1.stp"
FreeFormSurf2A = TesPartsDir / "FreeFormSurf2A.STEP"
FreeFormSurf3 = TesPartsDir / "FreeForm3.stp"
ConeTaped = TesPartsDir / "ConeTaped.stp"
FreeFormCLosed = TesPartsDir / "FreeFormClosed.stp"
Planar = TesPartsDir / "Planar.stp"
YachtBodypart  = TesPartsDir / "YachtBodypart.stp"
CircularSurf1  = TesPartsDir / "CircularSurf1.stp"
Cube           = TesPartsDir / "Cube.stp"
CircularSur2   = TesPartsDir / "CircularSur2.stp"
Conic          = TesPartsDir / "Conic.stp"
CircularHoles  = TesPartsDir / "CircularHoles.stp"
FullCylinder   = TesPartsDir / "FullCylinder.stp"
Sphere         = TesPartsDir / "Sphere.stp"
SphereTap      = TesPartsDir / "SphereTap.stp"
Tidebottle     = TesPartsDir / "Tidebottle.STEP"
FreeFormSurf4 = TesPartsDir / "FreeForm4.stp"
FreeFormBench = TesPartsDir / "FreeFormBench.stp"
FreeSharp1 = TesPartsDir / "FreeSharp1.stp"
FreeSharp2 = TesPartsDir / "FreeSharp2.stp"
SeatBr = TesPartsDir / "SeatBr.stp"
MouseBot = TesPartsDir / "MouseBot.stp"
CircularHoles05 = TesPartsDir / "CircularHoles05.stp"


shape_path = Planar

Face_Cad = CADTensorGenerator(
    device=device,
    seed_domain_mask_res=512,
    mesh_size_scale = 0.2,
)

cad_domain, face_mesh = Face_Cad.generate_from_file(shape_path)

dx,dy,dz = Face_Cad.print_face_info();



# In[ ]:


LoadingCase = "3PB_Planar"  # "Tensile" or "Compression"
voxel_size = float((dx + dy + dz) / 180.0)
thickness = 3.0 * voxel_size

common_settings = dict(
    thickness=thickness,
    voxel_size=voxel_size,
    extra_layers=1,
    tensors=face_mesh,

    voxelization_mode="triangle_distance",
    subvoxel_samples=1,
    min_geom_fraction=0.25,

    bc_mapping_mode="surface_patch",
    faces_index_base=0,
)

shell_problem = ThickenShell(
    **common_settings,

    load_case="fixed_side_loading",

    # Axis used to locate the two ends
    BC_dir="x",

    fixed_side="min",
    force_side="max",

    # Force direction
    load_dir="z",

    # "min" means negative Z
    load_direction_side="min",

    # Magnitude is converted to -10 N
    Load_magnitude=0.5,
)
loading_img = shell_problem.show_voxels_surface_and_bc(
    return_img=True,
    off_screen=True,
    window_size=(560, 430),
    show=True,
)

# In[4]:


path = "/home/arash/HVD_SeedsBase/Case_Studies/FreeForm3Seeds_60_New/optimized_shell_function.pt"
Explanation = "Seeds_plabbar_70_03thick_NEW"
Case_name = shape_path.stem
timelapse_output_folder=f"Case_Studies/{Case_name}{Explanation}"
viz = CADVisualizer()
fem =fem = run_fem_loss.NeuralTOMOFEM(shell_problem, device=device, isotropic=False)
cfg = TrainingConfig(
    LoadingCase=LoadingCase,

    #Start with pretrained
    warm_start_optimized_function_path=None,
    warm_start_prune_inactive_seeds=True,
    warm_start_skip_stage1=True,
    # Geometry and seed initialization
    seed_number=50,
    use_balanced_seed_init=True,
    seed_init_fps_seed=11,
    strut_thickness=0.4,
    Edge_in_losses="all",
    displacement_objective_mode="physical_max",

    # Seed separation
    min_seed_spacing=1.0,
    seed_spacing_power=2.0,
    lam_seed_spacing=1000.0,

    seed_spacing_safety_factor=1.05,
    seed_spacing_aggregate_temperature=0.005,

    seed_repulsion_factor=2,
    seed_repulsion_temperature_ratio=0.05,
    seed_repulsion_weight=2.0,

    # Decoder
    decoder_use_trim_activity=True,

    # Staged optimization
    eps=1e-12,
    scheduler_gamma=0.5,
    grad_clip_norm=2,
    log_every=100,

    # Decoder output
    use_3d_density_filter=False,
    generate_decoder_density_fiber=True,

    # Shared loss settings
    lam_cell_angle_uniform=0.5,
    lam_cell_radial_uniform=0.5,
    normalize_losses=True,

    # CVT
    cvt_temperature=0.01,
    curve_length_tolerance=0.10,
    curve_length_outlier_weight=5.0,

    # FEM
    fem_max_displacement=0.88,
    fem_yield_strength=400,
    fem_constraint_weight=1000.0,
    fem_stress_density_threshold=0.5,
    fem_training_safety_factor=0.90,
    fem_penal=3.0,
    fem_rho_min_ratio=1e-4,
    fem_rho_min_start=1e-3,
    fem_rho_min_end=1e-4,
    fem_density_floor=0.0,
    fem_baseline_weight=0.001,
    fem_violation_power=2.0,
    adaptive_fem_penalty=False,
    fem_lambda_initial=2.0,
    fem_lambda_min=2.0,
    fem_lambda_max=1e3,
    fem_lambda_growth=1.02,
    fem_lambda_decay=0.99,
    fem_constraint_tolerance=0.0,
    fem_safety_margin_weight=10.0,
    fem_constraint_p_norm = 12.0,

    # Invalid FEM handling
    skip_bad_fem_steps=True,
    invalid_fem_patience=3,
    invalid_lr_factor=0.5,
    minimum_learning_rate=1e-7,
    invalid_fem_consumes_budget=True,
    debug_fem_integrity=False,
    save_fem_debug_history=False,

    # Stage 1
    stage1_min_steps=1,
    stage1_max_steps=1000,
    stage1_patience=500,
    stage1_allow_seed_outside_domain=False,
    stage1_lam_fem=0.0,
    stage1_lam_cvt=5.0,
    stage1_lam_total_fiber_length=1.0,
    stage1_lam_l_curve_cell=1,
    stage1_scheduler_milestones=(0.5, 0.8),
    stage1_freeze_seeds=False,
    stage1_min_delta_abs=1e-4,
    stage1_min_delta_rel=1e-3,

    # Stage 2
    stage2_min_steps=3000,
    stage2_max_steps=10000,
    stage2_patience=300,
    stage2_allow_seed_outside_domain=True,  
    stage2_lam_fem=2,
    stage2_lam_cvt=0.5,
    stage2_lam_l_curve_cell=0,
    stage2_lam_total_fiber_length=2.0,
    stage2_scheduler_milestones=(0.8, 0.90),
    stage2_freeze_seeds=False,
    stage2_min_delta_abs=1e-4,
    stage2_min_delta_rel=1e-3,

    # Adaptive stage controller
    stage_topology_grace_steps=20,
    stage_topology_grace_max_resets=5,
    restore_transition_checkpoint=True,
    reset_optimizer_between_stages=True,
    stage_transition_selection="next_stage_objective",
    debug_stage_controller=False,

    # Learning rates
    lr_seed_refine=1e-4,
    lr_independent_seed_offsets=1e-4,
    lr_delta_head=1e-4,
    lr_mlp=1e-4,
    lr_decoder=1e-4,

    # Seed movement
    use_independent_seed_offsets=True,
    seed_offset_scale_start=0.1,
    seed_offset_scale_final=0.1,
    seed_offset_scale_ramp_frac=0.80,
    allow_seed_outside_domain=True,
    allow_seed_outside_domain_warmup_frac=0.01,
    use_rolling_seed_anchors=True,
    seed_anchor_warmup_frac=0.0,
    seed_anchor_momentum=0.05,
    anchor_guard_updates=False  ,
    disable_rolling_seed_anchors_for_curve_only=False,

    # Logging
    tensorboard_enabled=False,
    tensorboard_log_root="runs",
    experiment_name=f"{Case_name}{Explanation}",
    MakeTimelaps=True,
    timelapse_frame_step= 100,
    timelapse_output_folder=timelapse_output_folder,
)


trainer = NN_Trainer(
    generator=Face_Cad,
    viz=viz,
    decoder_cls=ContinuousVoronoiDecoder,
    ppnet_cls=PPNet,
    fem=fem,
    shell_problem=shell_problem,
    config=cfg,
    face_mesh=face_mesh,
)

result = trainer.train(
    shape_path=shape_path,
    face_tensors=[face_mesh],
)

print("best step:", result["best_step"])
print("best score:", result["best_score"])
print("density:", result["Final_shape_density"].shape)
print("fiber:", result["Final_shape_fiber_direction"].shape)
print("seed UV:", result["best_seed_points_uv"].shape)
print("seed XYZ:", result["best_seed_points_xyz"].shape)
