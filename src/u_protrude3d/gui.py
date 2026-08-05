"""Tkinter GUI for interactively editing u-Protrude3D config dataclasses."""
from __future__ import annotations

import dataclasses
import json
import tkinter as tk
from tkinter import filedialog, ttk
from typing import Any

from .config import (
    BenchmarkConfig,
    LargePatchConfig,
    SegmentConfig,
    StdPatchConfig,
    VolumeConfig,
    cMCFConfig,
    InitialHeightConfig,
)

_LITERAL_OPTIONS = {
    "H_segment_method": ("mean", "multiotsu"),
    "second_seg_H_method": ("mean", "multiotsu"),
    "ridge_segment_method": ("mean", "multiotsu"),
}

_PADDING = {"padx": 4, "pady": 2}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_scrollable_frame(parent: tk.Widget) -> tuple[tk.Canvas, ttk.Frame]:
    canvas = tk.Canvas(parent, borderwidth=0, highlightthickness=0)
    scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    inner = ttk.Frame(canvas)
    inner.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
    )
    canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    # Mouse-wheel scrolling
    canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", lambda ev: canvas.yview_scroll(int(-1 * (ev.delta / 120)), "units")))
    canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
    return canvas, inner


def _make_var(value: Any, name: str) -> tk.Variable:
    if isinstance(value, bool):
        v = tk.BooleanVar(value=value)
    elif isinstance(value, int):
        v = tk.StringVar(value=str(value))
    elif isinstance(value, float):
        v = tk.StringVar(value=str(value))
    elif isinstance(value, list):
        v = tk.StringVar(value=", ".join(str(x) for x in value))
    else:
        v = tk.StringVar(value=str(value))
    return v


def _add_param_row(
    frame: ttk.Frame,
    row: int,
    attr: str,
    value: Any,
    vars_map: dict,
    path: tuple,
) -> None:
    label = ttk.Label(frame, text=attr, anchor="w", width=32)
    label.grid(row=row, column=0, sticky="w", **_PADDING)

    var = _make_var(value, attr)
    vars_map[path + (attr,)] = (var, type(value))

    if isinstance(value, bool):
        w = ttk.Checkbutton(frame, variable=var)
        w.grid(row=row, column=1, sticky="w", **_PADDING)
    elif attr in _LITERAL_OPTIONS:
        w = ttk.Combobox(frame, textvariable=var, values=_LITERAL_OPTIONS[attr], state="readonly", width=16)
        w.grid(row=row, column=1, sticky="ew", **_PADDING)
    elif isinstance(value, list):
        w = ttk.Entry(frame, textvariable=var, width=40)
        w.grid(row=row, column=1, sticky="ew", **_PADDING)
        _attach_list_validation(w, var)
    elif isinstance(value, (int, float)):
        w = ttk.Entry(frame, textvariable=var, width=18)
        w.grid(row=row, column=1, sticky="ew", **_PADDING)
        _attach_numeric_validation(w, var, type(value))
    else:
        w = ttk.Entry(frame, textvariable=var, width=18)
        w.grid(row=row, column=1, sticky="ew", **_PADDING)

    frame.columnconfigure(1, weight=1)


def _attach_numeric_validation(entry: ttk.Entry, var: tk.StringVar, typ: type) -> None:
    def _check(*_):
        val = var.get()
        try:
            typ(val)
            entry.configure(style="TEntry")
        except (ValueError, tk.TclError):
            entry.configure(style="Error.TEntry")
    var.trace_add("write", _check)


def _attach_list_validation(entry: ttk.Entry, var: tk.StringVar) -> None:
    def _check(*_):
        try:
            vals = [float(x.strip()) for x in var.get().split(",") if x.strip()]
            if not all(0.0 <= v <= 1.0 for v in vals):
                raise ValueError
            entry.configure(style="TEntry")
        except (ValueError, tk.TclError):
            entry.configure(style="Error.TEntry")
    var.trace_add("write", _check)


def _build_fields_in_frame(
    frame: ttk.Frame,
    obj: Any,
    vars_map: dict,
    path: tuple,
    exclude: set | None = None,
) -> None:
    exclude = exclude or set()
    row = 0
    for f in dataclasses.fields(obj):
        if f.name in exclude:
            continue
        val = getattr(obj, f.name)
        if dataclasses.is_dataclass(val):
            continue
        _add_param_row(frame, row, f.name, val, vars_map, path)
        row += 1


def _build_labelframe(parent: ttk.Frame, title: str, obj: Any, fields: list[str], vars_map: dict, path: tuple) -> ttk.LabelFrame:
    lf = ttk.LabelFrame(parent, text=title, padding=4)
    row = 0
    for name in fields:
        val = getattr(obj, name)
        _add_param_row(lf, row, name, val, vars_map, path)
        row += 1
    lf.columnconfigure(1, weight=1)
    return lf


# ---------------------------------------------------------------------------
# Tab builders
# ---------------------------------------------------------------------------

def _build_segment_tab(notebook: ttk.Notebook, cfg: SegmentConfig, vars_map: dict) -> None:
    outer = ttk.Frame(notebook)
    notebook.add(outer, text="Segment")

    inner_nb = ttk.Notebook(outer)
    inner_nb.pack(fill="both", expand=True)

    # --- General sub-tab ---
    gen_outer = ttk.Frame(inner_nb)
    inner_nb.add(gen_outer, text="General")
    _, gen_frame = _make_scrollable_frame(gen_outer)

    flat_fields = [f.name for f in dataclasses.fields(cfg) if not dataclasses.is_dataclass(getattr(cfg, f.name))]
    mesh_group = ["voxel_size", "min_size_comps_initial", "min_size_comps_protrude_patch",
                  "sdf_binary_dilate_ksize", "sdf_binary_erode_ksize", "curvature_radius"]
    smooth_group = ["n_smooth_scalar_fn_iters", "offset_ref_ind"]
    vis_group = ["n_protrude_colors", "random_seed", "debug_viz"]

    r = 0
    lf1 = _build_labelframe(gen_frame, "Mesh processing", cfg, [f for f in mesh_group if f in flat_fields], vars_map, ("segment",))
    lf1.grid(row=r, column=0, sticky="ew", **_PADDING); r += 1
    lf2 = _build_labelframe(gen_frame, "Smoothing & scale", cfg, [f for f in smooth_group if f in flat_fields], vars_map, ("segment",))
    lf2.grid(row=r, column=0, sticky="ew", **_PADDING); r += 1
    lf3 = _build_labelframe(gen_frame, "Visualisation", cfg, [f for f in vis_group if f in flat_fields], vars_map, ("segment",))
    lf3.grid(row=r, column=0, sticky="ew", **_PADDING)
    gen_frame.columnconfigure(0, weight=1)

    # --- cMCF sub-tab ---
    cmcf_outer = ttk.Frame(inner_nb)
    inner_nb.add(cmcf_outer, text="cMCF")
    _, cmcf_frame = _make_scrollable_frame(cmcf_outer)
    _build_fields_in_frame(cmcf_frame, cfg.cmcf, vars_map, ("segment", "cmcf"))

    # --- Initial Height sub-tab ---
    ih_outer = ttk.Frame(inner_nb)
    inner_nb.add(ih_outer, text="Initial Height")
    _, ih_frame = _make_scrollable_frame(ih_outer)
    _build_fields_in_frame(ih_frame, cfg.initial_height, vars_map, ("segment", "initial_height"))

    # --- Large Patch sub-tab ---
    lp_outer = ttk.Frame(inner_nb)
    inner_nb.add(lp_outer, text="Large Patch")
    _, lp_frame = _make_scrollable_frame(lp_outer)

    r = 0
    lp_area = _build_labelframe(lp_frame, "Area thresholds", cfg.large_patch, ["min_max_area", "max_area_thresh_factor", "curv_ridge_ratio_threshold", "occ_threshold"], vars_map, ("segment", "large_patch"))
    lp_area.grid(row=r, column=0, sticky="ew", **_PADDING); r += 1
    lp_h = _build_labelframe(lp_frame, "H segmentation", cfg.large_patch, ["H_segment_method", "H_segment_otsu_n_levels", "H_segment_otsu_level", "second_seg_H_method", "second_seg_H_otsu_n_levels", "second_seg_H_otsu_level"], vars_map, ("segment", "large_patch"))
    lp_h.grid(row=r, column=0, sticky="ew", **_PADDING); r += 1
    lp_r = _build_labelframe(lp_frame, "Ridge segmentation", cfg.large_patch, ["ridge_segment_method", "ridge_otsu_n_levels", "ridge_otsu_level"], vars_map, ("segment", "large_patch"))
    lp_r.grid(row=r, column=0, sticky="ew", **_PADDING)
    lp_frame.columnconfigure(0, weight=1)

    # --- Std Patch sub-tab ---
    sp_outer = ttk.Frame(inner_nb)
    inner_nb.add(sp_outer, text="Std Patch")
    _, sp_frame = _make_scrollable_frame(sp_outer)

    r = 0
    sp_thresh = _build_labelframe(sp_frame, "Ridgeness thresholds", cfg.std_patch, ["curv_ridge_ratio_threshold"], vars_map, ("segment", "std_patch"))
    sp_thresh.grid(row=r, column=0, sticky="ew", **_PADDING); r += 1
    sp_h = _build_labelframe(sp_frame, "H segmentation", cfg.std_patch, ["H_segment_method", "H_segment_otsu_n_levels", "H_segment_otsu_level", "H_segment_erode_steps", "H_segment_use_local_adaptive", "H_local_adaptive_smooth_iters", "apply_power_H_correct", "power_H_correct"], vars_map, ("segment", "std_patch"))
    sp_h.grid(row=r, column=0, sticky="ew", **_PADDING); r += 1
    sp_r = _build_labelframe(sp_frame, "Ridge segmentation", cfg.std_patch, ["ridge_segment_method", "ridge_otsu_n_levels", "ridge_otsu_level", "ridge_segment_erode_steps", "ridge_use_local_adaptive", "ridge_local_adaptive_smooth_iters", "apply_power_ridge_correct", "power_ridge_correct"], vars_map, ("segment", "std_patch"))
    sp_r.grid(row=r, column=0, sticky="ew", **_PADDING); r += 1
    sp_plan = _build_labelframe(sp_frame, "Planarity & multi-bleb", cfg.std_patch, ["planarity_check_frac", "planarity_check_dilate_binary", "multibleb_min_recovered_frac", "multibleb_min_occ_area_fraction", "multibleb_max_mean_aspect_ratio", "multibleb_check_neck_neg_curvature", "multibleb_neck_neg_curvature_thresh"], vars_map, ("segment", "std_patch"))
    sp_plan.grid(row=r, column=0, sticky="ew", **_PADDING)
    sp_frame.columnconfigure(0, weight=1)


def _build_benchmark_tab(notebook: ttk.Notebook, cfg: BenchmarkConfig, vars_map: dict) -> None:
    outer = ttk.Frame(notebook)
    notebook.add(outer, text="Benchmark")
    _, frame = _make_scrollable_frame(outer)
    _build_fields_in_frame(frame, cfg, vars_map, ("benchmark",))


def _build_volume_tab(notebook: ttk.Notebook, cfg: VolumeConfig, vars_map: dict) -> None:
    outer = ttk.Frame(notebook)
    notebook.add(outer, text="Volume")
    _, frame = _make_scrollable_frame(outer)
    _build_fields_in_frame(frame, cfg, vars_map, ("volume",))


# ---------------------------------------------------------------------------
# Apply / reset / JSON
# ---------------------------------------------------------------------------

def _resolve_path(configs: dict, path: tuple) -> tuple[Any, str]:
    """Return (parent_object, attr_name) for the given dotted path tuple."""
    obj = configs[path[0]]
    for part in path[1:-1]:
        obj = getattr(obj, part)
    return obj, path[-1]


def _coerce(var: tk.Variable, typ: type) -> Any:
    raw = var.get()
    if typ is bool:
        return bool(raw)
    if typ is int:
        return int(raw)
    if typ is float:
        return float(raw)
    if typ is list:
        return [float(x.strip()) for x in raw.split(",") if x.strip()]
    return str(raw)


def _apply(vars_map: dict, configs: dict, status_label: ttk.Label) -> None:
    errors = []
    for path, (var, typ) in vars_map.items():
        try:
            value = _coerce(var, typ)
            obj, attr = _resolve_path(configs, path)
            setattr(obj, attr, value)
        except Exception as exc:
            errors.append(f"{'.'.join(path)}: {exc}")
    if errors:
        status_label.configure(text="Error: " + "; ".join(errors), foreground="red")
    else:
        status_label.configure(text="Applied", foreground="green")


def _reset(vars_map: dict, configs: dict, status_label: ttk.Label) -> None:
    defaults = {
        "segment": SegmentConfig(),
        "benchmark": BenchmarkConfig(),
        "volume": VolumeConfig(),
    }
    for path, (var, typ) in vars_map.items():
        obj, attr = _resolve_path(defaults, path)
        val = getattr(obj, attr)
        if typ is list:
            var.set(", ".join(str(x) for x in val))
        else:
            var.set(str(val) if not isinstance(val, bool) else val)
    # Write defaults back into configs
    for key in configs:
        configs[key].__dict__.update(defaults[key].__dict__)
    status_label.configure(text="Reset to defaults", foreground="blue")


def _export_json(configs: dict, status_label: ttk.Label) -> None:
    path = filedialog.asksaveasfilename(
        defaultextension=".json",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        title="Export config",
    )
    if not path:
        return
    data = {k: dataclasses.asdict(v) for k, v in configs.items()}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    status_label.configure(text=f"Exported to {path}", foreground="green")


def _load_json(vars_map: dict, configs: dict, status_label: ttk.Label) -> None:
    path = filedialog.askopenfilename(
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        title="Load config",
    )
    if not path:
        return
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        status_label.configure(text=f"Load error: {exc}", foreground="red")
        return

    # Reconstruct config objects from nested dicts
    def _from_dict(cls, d):
        kwargs = {}
        for f in dataclasses.fields(cls):
            if f.name not in d:
                continue
            if dataclasses.is_dataclass(f.type if isinstance(f.type, type) else type(None)):
                kwargs[f.name] = _from_dict(f.type, d[f.name])
            else:
                kwargs[f.name] = d[f.name]
        return cls(**kwargs)

    mapping = {"segment": SegmentConfig, "benchmark": BenchmarkConfig, "volume": VolumeConfig}
    for key, cls in mapping.items():
        if key in data:
            try:
                configs[key].__dict__.update(_from_dict(cls, data[key]).__dict__)
            except Exception:
                pass

    # Refresh vars from updated configs
    for path, (var, typ) in vars_map.items():
        try:
            obj, attr = _resolve_path(configs, path)
            val = getattr(obj, attr)
            if typ is list:
                var.set(", ".join(str(x) for x in val))
            else:
                var.set(str(val) if not isinstance(val, bool) else val)
        except Exception:
            pass
    status_label.configure(text=f"Loaded from {path}", foreground="green")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def launch_config_gui(
    cfg_segment: SegmentConfig | None = None,
    cfg_benchmark: BenchmarkConfig | None = None,
    cfg_volume: VolumeConfig | None = None,
) -> tuple[SegmentConfig, BenchmarkConfig, VolumeConfig]:
    """Open a Tkinter GUI for editing configs. Blocks until the window is closed.

    Returns the three (possibly modified) config objects.
    """
    cfg_segment = cfg_segment or SegmentConfig()
    cfg_benchmark = cfg_benchmark or BenchmarkConfig()
    cfg_volume = cfg_volume or VolumeConfig()

    configs = {"segment": cfg_segment, "benchmark": cfg_benchmark, "volume": cfg_volume}
    vars_map: dict = {}

    root = tk.Tk()
    root.title("u-Protrude3D Config Editor")
    root.minsize(560, 480)

    # Error entry style
    style = ttk.Style(root)
    style.configure("Error.TEntry", fieldbackground="#ffcccc")

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=6, pady=6)

    _build_segment_tab(notebook, cfg_segment, vars_map)
    _build_benchmark_tab(notebook, cfg_benchmark, vars_map)
    _build_volume_tab(notebook, cfg_volume, vars_map)

    # Bottom toolbar
    toolbar = ttk.Frame(root)
    toolbar.pack(fill="x", padx=6, pady=(0, 6))

    status = ttk.Label(toolbar, text="", anchor="w")
    status.pack(side="left", fill="x", expand=True)

    ttk.Button(toolbar, text="Load JSON", command=lambda: _load_json(vars_map, configs, status)).pack(side="right", padx=2)
    ttk.Button(toolbar, text="Export JSON", command=lambda: _export_json(configs, status)).pack(side="right", padx=2)
    ttk.Button(toolbar, text="Reset to defaults", command=lambda: _reset(vars_map, configs, status)).pack(side="right", padx=2)
    ttk.Button(toolbar, text="Apply", command=lambda: _apply(vars_map, configs, status)).pack(side="right", padx=2)

    root.mainloop()

    # Attempt a final apply in case the user closed without clicking Apply
    try:
        _apply(vars_map, configs, status)
    except Exception:
        pass

    return cfg_segment, cfg_benchmark, cfg_volume
