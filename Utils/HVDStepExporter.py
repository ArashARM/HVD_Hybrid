from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class HVDStepExportSummary:
    output_path: str
    curve_count: int
    segment_count: int
    radius: float
    fused: bool


class HVDStepExporter:
    """
    Export an optimized HVD curve network as a STEP B-rep tube model.

    The optimizer result returned by NN_Trainer contains enough geometry for
    this exporter: best_edge_curves_xyz plus either centerline_radius or
    strut_thickness. The exported STEP is intended for Abaqus import/testing.
    """

    def __init__(
        self,
        *,
        default_radius: float | None = None,
        include_edge_types: Iterable[int] | None = None,
        fuse: bool = False,
        add_joint_spheres: bool = True,
        min_segment_length: float = 1.0e-9,
    ):
        self.default_radius = None if default_radius is None else float(default_radius)
        self.include_edge_types = (
            None if include_edge_types is None else {int(v) for v in include_edge_types}
        )
        self.fuse = bool(fuse)
        self.add_joint_spheres = bool(add_joint_spheres)
        self.min_segment_length = float(min_segment_length)

    @staticmethod
    def _np():
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError(
                "HVD STEP export requires numpy. Install the project's CAD/export "
                "environment dependencies, then rerun HVDStepExporter."
            ) from exc
        return np

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        np = HVDStepExporter._np()
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @classmethod
    def curves_from_optimization_output(cls, optimization_output: dict[str, Any]) -> np.ndarray:
        curves = optimization_output.get("best_edge_curves_xyz", None)
        if curves is None and optimization_output.get("best_pred"):
            curves = optimization_output["best_pred"][0].get("edge_curves_xyz", None)
        if curves is None:
            raise ValueError(
                "Optimization output does not contain best_edge_curves_xyz or "
                "best_pred[0]['edge_curves_xyz']."
            )
        curves_np = cls._to_numpy(curves).astype(float)
        if curves_np.ndim != 3 or curves_np.shape[-1] != 3:
            raise ValueError(
                "edge_curves_xyz must have shape [num_edges, samples_per_edge, 3], "
                f"got {curves_np.shape}."
            )
        return curves_np

    @classmethod
    def edge_types_from_optimization_output(
        cls,
        optimization_output: dict[str, Any],
    ) -> np.ndarray | None:
        edge_type = optimization_output.get("best_edge_type", None)
        if edge_type is None:
            graph = optimization_output.get("best_graph", None)
            if isinstance(graph, dict):
                edge_type = graph.get("edge_type", None)
        if edge_type is None and optimization_output.get("best_pred"):
            pred = optimization_output["best_pred"][0]
            edge_type = pred.get("edge_type", None)
            if edge_type is None and isinstance(pred.get("graph"), dict):
                edge_type = pred["graph"].get("edge_type", None)
        if edge_type is None:
            return None
        return cls._to_numpy(edge_type).astype(int).reshape(-1)

    @classmethod
    def radius_from_optimization_output(
        cls,
        optimization_output: dict[str, Any],
        default_radius: float | None = None,
    ) -> float:
        candidates = [
            optimization_output.get("centerline_radius", None),
            optimization_output.get("best_centerline_radius", None),
        ]
        if optimization_output.get("best_pred"):
            candidates.append(optimization_output["best_pred"][0].get("centerline_radius", None))
        for value in candidates:
            if value is None:
                continue
            np = cls._np()
            radius = float(np.asarray(cls._to_numpy(value), dtype=float).mean())
            if np.isfinite(radius) and radius > 0.0:
                return radius

        strut_thickness = optimization_output.get("strut_thickness", None)
        if strut_thickness is not None:
            np = cls._np()
            radius = 0.5 * float(strut_thickness)
            if np.isfinite(radius) and radius > 0.0:
                return radius

        if default_radius is not None and float(default_radius) > 0.0:
            return float(default_radius)
        raise ValueError(
            "Could not infer a positive tube radius. Pass default_radius=... "
            "or include centerline_radius/strut_thickness in the optimization output."
        )

    def export_optimization_output(
        self,
        optimization_output: dict[str, Any],
        output_path: str | Path,
        *,
        radius: float | None = None,
    ) -> HVDStepExportSummary:
        curves = self.curves_from_optimization_output(optimization_output)
        edge_types = self.edge_types_from_optimization_output(optimization_output)
        export_radius = (
            float(radius)
            if radius is not None
            else self.radius_from_optimization_output(optimization_output, self.default_radius)
        )
        return self.export_curves(
            curves,
            output_path,
            radius=export_radius,
            edge_types=edge_types,
        )

    def export_curves(
        self,
        curves_xyz: Any,
        output_path: str | Path,
        *,
        radius: float,
        edge_types: Any | None = None,
    ) -> HVDStepExportSummary:
        np = self._np()
        occ = self._load_occ()
        curves = self._to_numpy(curves_xyz).astype(float)
        if curves.ndim != 3 or curves.shape[-1] != 3:
            raise ValueError(f"curves_xyz must have shape [E, K, 3], got {curves.shape}.")
        if not np.isfinite(curves).all():
            curves = np.where(np.isfinite(curves), curves, np.nan)
        edge_types_np = None if edge_types is None else self._to_numpy(edge_types).astype(int).reshape(-1)

        shapes = []
        segment_count = 0
        kept_curve_count = 0
        for edge_id, curve in enumerate(curves):
            if edge_types_np is not None and edge_id < edge_types_np.size:
                if self.include_edge_types is not None and int(edge_types_np[edge_id]) not in self.include_edge_types:
                    continue
            curve = self._clean_curve_points(curve)
            if curve.shape[0] < 2:
                continue
            kept_curve_count += 1
            curve_shapes, curve_segments = self._tube_shapes_for_curve(
                curve,
                float(radius),
                occ,
            )
            shapes.extend(curve_shapes)
            segment_count += curve_segments

        if not shapes:
            raise ValueError("No valid curve segments were available for STEP export.")

        shape = self._fuse_shapes(shapes, occ) if self.fuse else self._compound_shapes(shapes, occ)
        output_path = str(Path(output_path))
        self._write_step(shape, output_path, occ)
        return HVDStepExportSummary(
            output_path=output_path,
            curve_count=kept_curve_count,
            segment_count=segment_count,
            radius=float(radius),
            fused=self.fuse,
        )

    def _clean_curve_points(self, curve: np.ndarray) -> np.ndarray:
        np = self._np()
        curve = np.asarray(curve, dtype=float)
        curve = curve[np.isfinite(curve).all(axis=1)]
        if curve.shape[0] < 2:
            return curve
        keep = [0]
        for idx in range(1, curve.shape[0]):
            if np.linalg.norm(curve[idx] - curve[keep[-1]]) > self.min_segment_length:
                keep.append(idx)
        return curve[np.asarray(keep, dtype=int)]

    def _tube_shapes_for_curve(self, curve: np.ndarray, radius: float, occ: dict[str, Any]):
        np = self._np()
        shapes = []
        segment_count = 0
        for a, b in zip(curve[:-1], curve[1:]):
            direction = b - a
            length = float(np.linalg.norm(direction))
            if length <= self.min_segment_length:
                continue
            axis = occ["gp_Ax2"](
                occ["gp_Pnt"](float(a[0]), float(a[1]), float(a[2])),
                occ["gp_Dir"](float(direction[0]), float(direction[1]), float(direction[2])),
            )
            shapes.append(occ["BRepPrimAPI_MakeCylinder"](axis, float(radius), length).Shape())
            segment_count += 1

        if self.add_joint_spheres:
            for point in curve:
                shapes.append(
                    occ["BRepPrimAPI_MakeSphere"](
                        occ["gp_Pnt"](float(point[0]), float(point[1]), float(point[2])),
                        float(radius),
                    ).Shape()
                )
        return shapes, segment_count

    @staticmethod
    def _load_occ() -> dict[str, Any]:
        try:
            from OCC.Core.BRep import BRep_Builder
            from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
            from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeSphere
            from OCC.Core.IFSelect import IFSelect_RetDone
            from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer
            from OCC.Core.TopoDS import TopoDS_Compound
            from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt
        except ImportError as exc:
            raise ImportError(
                "STEP export requires pythonocc-core/OpenCascade. Install it in the "
                "environment used for CAD export, then rerun HVDStepExporter."
            ) from exc

        return {
            "BRep_Builder": BRep_Builder,
            "BRepAlgoAPI_Fuse": BRepAlgoAPI_Fuse,
            "BRepPrimAPI_MakeCylinder": BRepPrimAPI_MakeCylinder,
            "BRepPrimAPI_MakeSphere": BRepPrimAPI_MakeSphere,
            "IFSelect_RetDone": IFSelect_RetDone,
            "STEPControl_AsIs": STEPControl_AsIs,
            "STEPControl_Writer": STEPControl_Writer,
            "TopoDS_Compound": TopoDS_Compound,
            "gp_Ax2": gp_Ax2,
            "gp_Dir": gp_Dir,
            "gp_Pnt": gp_Pnt,
        }

    @staticmethod
    def _compound_shapes(shapes: list[Any], occ: dict[str, Any]):
        compound = occ["TopoDS_Compound"]()
        builder = occ["BRep_Builder"]()
        builder.MakeCompound(compound)
        for shape in shapes:
            builder.Add(compound, shape)
        return compound

    @staticmethod
    def _fuse_shapes(shapes: list[Any], occ: dict[str, Any]):
        fused = shapes[0]
        for shape in shapes[1:]:
            fuse_op = occ["BRepAlgoAPI_Fuse"](fused, shape)
            fuse_op.Build()
            if not fuse_op.IsDone():
                raise RuntimeError("OpenCascade boolean fuse failed during STEP export.")
            fused = fuse_op.Shape()
        return fused

    @staticmethod
    def _write_step(shape: Any, output_path: str, occ: dict[str, Any]) -> None:
        writer = occ["STEPControl_Writer"]()
        writer.Transfer(shape, occ["STEPControl_AsIs"])
        status = writer.Write(output_path)
        if status != occ["IFSelect_RetDone"]:
            raise RuntimeError(f"OpenCascade failed to write STEP file: {output_path}")
