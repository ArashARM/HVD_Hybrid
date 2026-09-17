import torch

from Decoder_CLasses.ContinuousVoronoiDecoder import ContinuousVoronoiDecoder
from Decoder_CLasses.NearUniformHoneycombBaseline import NearUniformHoneycombBaseline


def generate_near_uniform_honeycomb_fields(Face_Cad, face_mesh, fem=None, device="cpu"):
    decoder = ContinuousVoronoiDecoder(
        Cad_domain=Face_Cad,
        face_mesh=face_mesh,
        strut_thickness=0.25,
    )
    baseline = NearUniformHoneycombBaseline(
        decoder=decoder,
        cad_domain=Face_Cad,
        face_mesh=face_mesh,
        target_seed_count=50,
        spacing_uv=None,
        mode="uv_hex",
    )

    output = baseline.generate(
        device=torch.device(device),
        dtype=torch.float64,
        generate_density_fiber=True,
    )

    rho_surface = output["rho"]
    fiber_surface = output["fiber3d"]

    if fem is not None:
        # Existing shell/FEM wrappers can map these surface fields to the FEM
        # element fields expected by the solver.
        fem_fields = fem.problem.surface_fields_to_element_fields(
            rho_surface,
            fiber_surface,
        )
        return output, fem_fields

    return output
