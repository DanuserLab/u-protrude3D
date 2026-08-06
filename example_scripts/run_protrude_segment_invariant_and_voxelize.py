# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 16:55:18 2026

@author: fyz11
"""


if __name__=="__main__":
    import numpy as np 
    import pylab as plt 
    import scipy.io as spio 
    import skimage.io as skio 
    import os 
    import glob 
    import scipy.ndimage as ndimage 
    
    
    import skimage.segmentation as sksegmentation 
    
    import unwrap3D.Utility_Functions.file_io as fio 
    import unwrap3D.Mesh.meshtools as meshtools
    
    
    import skimage.measure as skmeasure 
    import skimage.exposure as skexposure
    import skimage.morphology as skmorph 
    
    
    """
    add the u-protrude3D package - ignore if installed.
    """
    from pathlib import Path
    import sys
    
    # --- make the package importable when run from the repo without install ------
    _SRC = Path(__file__).resolve().parent / "src"
    if _SRC.is_dir():
        sys.path.insert(0, str(_SRC))
    
    import u_protrude3d as up3d
    
    

    """
    1. u-Protrude3D settings for detection
    """
    cfg = up3d.SegmentConfig()
    
    # cMCF inflation (cfg.cmcf.*)
    cfg.cmcf.n_iters = 20          # inflation steps — more = taller height field
    cfg.cmcf.step_size=0.5
    cfg.cmcf.solver = 'pardiso'    # or 'scipy' without MKL
    
    # Initial height binarization (cfg.initial_height.*)
    cfg.initial_height.use_auto = True
    cfg.initial_height.use_mean = True   # mean threshold (vs use_otsu=True) # if True, otsu won't run. 
    # cfg.initial_height.use_otsu = True
    cfg.initial_height.otsu_n_levels = 3
    cfg.initial_height.otsu_level = -1
    cfg.initial_height.prop_iters = 1
    cfg.initial_height.prop_rebinarize = 0.25
    
    # Initial CC filtering
    cfg.min_size_comps_initial = 5          # min faces for a CC to survive
    cfg.min_size_comps_protrude_patch = 5    # min faces for a sub-CC seed
    # cfg.n_smooth_scalar_fn_iters = 100        # smoothing of height/curvature fields (was 100)
    # cfg.n_smooth_scalar_fn_iters = 50
    
    # Large patch splitting (cfg.large_patch.*)
    cfg.large_patch.min_max_area = 2500     # face-count threshold for "large" patch
    cfg.large_patch.max_area_thresh_factor = 0.0
    
    # SI-based seeding (cfg.invariant.*)
    cfg.invariant.si_segment_method = 'multiotsu'   # or 'mean'
    cfg.invariant.si_segment_otsu_n_levels = 2
    cfg.invariant.si_segment_otsu_level = -1
    cfg.invariant.n_diffusion_iters = 5
    
    
    # adaptive-merging and splitting path
    cfg.invariant.ws_saddle_criterion = 'adaptive'
    # cfg.invariant.ws_saddle_ar_threshold = 3.0   # tune down to 1.5 if you want stricter "compact" definition
    
    # ok... these gates r working ish.
    cfg.invariant.ws_saddle_ar_threshold = 0.3 # lower encourages bleb merging i.e. using si_threshold
    cfg.invariant.ws_saddle_depth_threshold = 0.2  # used for elongated pairs
    cfg.invariant.ws_saddle_si_threshold = 0.3     # used for compact pairs
        
    

    """
    Specify paths to mesh file, and the save out directory  
    """
    # meshfile = '../example_data/synthetic_data/bleb/Cell101/mesh_gt_protrusion_color.obj'
    # saveoout = os.path.join(r'D:\Work\Projects\Danuser\u-Protrude3D_paper', 
    #                         'example_data/synthetic_data/bleb/Cell101/',
    #                         'segment_protrusion_invariant_merge')
    
    meshfile = '../example_data/synthetic_data/ruffle/Cell001/mesh_gt_protrusion_color.obj'
    saveoout = os.path.join(r'D:\Work\Projects\Danuser\u-Protrude3D_paper', 
                            'example_data/synthetic_data/ruffle/Cell001/',
                            'segment_protrusion_invariant_merge')
    
    
    print("\n[1] segment_protrusions ...")
    res = up3d.segment_protrusions_invariant(mesh_path = meshfile, 
                                             save_dir = saveoout,
                                             cfg=cfg)
    pred_labels = res.vertex_labels


    
    """
    2. u-Protrude3D average precision benchmarking of vertex segmentation labels (if ground truth .mat available)
    """
    # gt_dir = '../example_data/synthetic_data/bleb/Cell101'
    # pred_dir = saveoout
    # benchmarkout = os.path.join(r'D:\Work\Projects\Danuser\u-Protrude3D_paper', 
    #                         'example_data/synthetic_data/bleb/Cell101/',
    #                         'segment_protrusion_invariant_merge',
                            # 'benchmark')
    
    gt_dir = '../example_data/synthetic_data/ruffle/Cell001'
    pred_dir = saveoout
    benchmarkout = os.path.join(r'D:\Work\Projects\Danuser\u-Protrude3D_paper', 
                            'example_data/synthetic_data/ruffle/Cell001/',
                            'segment_protrusion_invariant_merge',
                            'benchmark')
    
    bench = up3d.benchmark_segmentation(
                                        pred_dirs=[pred_dir],          # folder containing instance_protrusion_segmentation_stats.mat
                                        gt_dirs=[gt_dir],                # folder containing protrusion_labels_GT_surface.mat
                                        pred_mesh_filename="inv_final_labels.obj",
                                        gt_mesh_filename="mesh_gt_protrusion_color.obj",
                                        pred_mat_filename="instance_protrusion_segmentation_stats.mat",  # default
                                        gt_mat_filename="protrusion_labels_GT_surface.mat",              # default
                                        save_dir=benchmarkout)
                                    
    print(bench.ap)              # AP curve over IoU thresholds
    print(bench.iou_thresholds)   
    print(bench.per_cell_names)  # cell identifiers
    
    # plot the curve
    plt.figure(figsize=(5,5))
    plt.plot(bench.iou_thresholds, 
             bench.ap[0], 
             lw=3, 
             color='k')
    plt.xlim([0.5,1])
    plt.ylim([0,1])
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.ylabel('Average Precision', fontsize=18)
    plt.xlabel('IoU cutoff', fontsize=18)
    plt.tick_params(length=10, right=True)
    plt.show()


    """
    3. u-Protrude3D voxelization of protrusions and 
    """
    
    # get the final segmented mesh
    protrude_segment_meshfile = os.path.join(saveoout, 
                                              "inv_final_labels.obj")
    # define the saveout of the voxelize. 
    saveoout_voxelize = os.path.join(saveoout, 
                                     'segment_protrusion_invariant_merge_voxelize')
    
    
    # --- GVF path with VFC enabled (experimental) ---
    cfg_voxelize = up3d.VolumeConfig()
    cfg_voxelize.mesh_sw_min_size = 10000
    cfg_voxelize.use_mesh_based_shrinkwrap = False   # always False to guarantee a good basal cortex wrap. mesh currently cannot handle very large holes.
    
    cfg_voxelize.gvf_mu = 0.01
    cfg_voxelize.gvf_iters = 15
    cfg_voxelize.gvf_vfc_sigma = 1.0   # enable VFC on the GVF path (off by default) 
    cfg_voxelize.gvf_vfc_blend = 0.0   # blend GVF and VFC forces equally
    # cfg.mesh_sw_vfc_sigma = 2.0    # VFC field Gaussian sigma (larger = smoother force field)
    # cfg.mesh_sw_vfc_blend = 0.5

    
    cfg_voxelize.mesh_sw_genus0_alpha_auto = False
    # cfg_voxelize.mesh_sw_force_sigma = 6.0
    cfg_voxelize.mesh_sw_fill_holes = True # this is necessary.
    

    vol = up3d.volumize_protrusions(protrude_segment_meshfile,
                                    res.vertex_labels,
                                    tif_path = None, # not used. 
                                    save_dir=saveoout_voxelize,
                                    cMCF_steps=None,
                                    cfg = cfg_voxelize)
    
















