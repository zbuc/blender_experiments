"""
EXP-E: Multi-view Profile Fusion Experiment

This script tests different profile fusion strategies to improve IoU scores,
particularly for front/side views which are currently at 0.64-0.66.

Test Variations:
- E1: Equal weights (0.5, 0.5)
- E2: Front-heavy (0.6, 0.4)
- E3: Side-heavy (0.4, 0.6)
- E4: Adaptive weights based on profile confidence

Success Criteria:
- Front/Side IoU improves from 0.64-0.66 → 0.68-0.72
- Top view maintains 0.97 IoU (no regression)
- Average IoU: 0.75 → 0.79-0.80

Expected Impact: +2-5% IoU (High confidence)
Baseline: 0.75 IoU average
"""

import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import required modules
try:
    import bpy
    from main_integration import BlockingWorkflow
    from integration.shape_matching.shape_matcher import calculate_iou, render_silhouette_from_mesh
    BLENDER_AVAILABLE = True
except ImportError as e:
    print(f"Error: {e}")
    print("This script must be run in Blender:")
    print("  blender --background --python test_profile_fusion_exp.py")
    sys.exit(1)


def run_experiment(strategy_name, fusion_strategy, front_path, side_path, top_path, num_slices=12):
    """
    Run a single profile fusion experiment.

    Args:
        strategy_name: Name of the experiment (E1, E2, E3, E4)
        fusion_strategy: Fusion strategy to use
        front_path: Path to front reference image
        side_path: Path to side reference image
        top_path: Path to top reference image
        num_slices: Number of vertical slices

    Returns:
        Dictionary with results
    """
    print(f"\n{'='*70}")
    print(f"Running {strategy_name}: {fusion_strategy} fusion")
    print(f"{'='*70}")

    # Create workflow
    workflow = BlockingWorkflow(
        front_path=front_path,
        side_path=side_path,
        top_path=top_path
    )

    # Run workflow with specified fusion strategy
    final_mesh = workflow.run_full_workflow(
        num_slices=num_slices,
        profile_fusion_strategy=fusion_strategy
    )

    if not final_mesh:
        print(f"  ERROR: Failed to create mesh for {strategy_name}")
        return None

    # Calculate IoU scores for each view
    print(f"\n{strategy_name}: Calculating IoU scores...")
    results = {
        'strategy_name': strategy_name,
        'fusion_strategy': fusion_strategy,
        'iou_scores': {}
    }

    # Load reference images
    import cv2
    import numpy as np
    from integration.shape_matching.profile_extractor import extract_silhouette_from_image

    ref_images = {}
    ref_silhouettes = {}

    if front_path:
        ref_images['front'] = cv2.imread(front_path)
        if ref_images['front'] is not None:
            ref_silhouettes['front'] = extract_silhouette_from_image(ref_images['front'])

    if side_path:
        ref_images['side'] = cv2.imread(side_path)
        if ref_images['side'] is not None:
            ref_silhouettes['side'] = extract_silhouette_from_image(ref_images['side'])

    if top_path:
        ref_images['top'] = cv2.imread(top_path)
        if ref_images['top'] is not None:
            ref_silhouettes['top'] = extract_silhouette_from_image(ref_images['top'])

    # Render mesh from each orthogonal view and calculate IoU
    view_angles = {
        'front': (90, 0, 0),   # Look from front (X-axis)
        'side': (90, 0, 90),   # Look from side (Y-axis)
        'top': (0, 0, 0)       # Look from top (Z-axis)
    }

    for view_name, angles in view_angles.items():
        if view_name in ref_silhouettes:
            try:
                # Render mesh silhouette from this view
                rendered_silhouette = render_silhouette_from_mesh(
                    final_mesh,
                    camera_angle=angles,
                    resolution=(ref_silhouettes[view_name].shape[1], ref_silhouettes[view_name].shape[0])
                )

                # Calculate IoU
                iou = calculate_iou(ref_silhouettes[view_name], rendered_silhouette)
                results['iou_scores'][view_name] = iou
                print(f"  {view_name.upper()} IoU: {iou:.4f}")
            except Exception as e:
                print(f"  ERROR: Could not calculate IoU for {view_name}: {e}")
                results['iou_scores'][view_name] = None

    # Calculate average IoU
    valid_ious = [iou for iou in results['iou_scores'].values() if iou is not None]
    if valid_ious:
        results['average_iou'] = sum(valid_ious) / len(valid_ious)
        print(f"\n  AVERAGE IoU: {results['average_iou']:.4f}")
    else:
        results['average_iou'] = None

    # Clean up mesh
    bpy.data.objects.remove(final_mesh, do_unlink=True)

    return results


def run_all_experiments(front_path=None, side_path=None, top_path=None, num_slices=12):
    """
    Run all profile fusion experiments (E1-E4).

    Args:
        front_path: Path to front reference image
        side_path: Path to side reference image
        top_path: Path to top reference image
        num_slices: Number of vertical slices

    Returns:
        List of results dictionaries
    """
    print("\n" + "="*70)
    print("EXP-E: Multi-view Profile Fusion Experiments")
    print("="*70)
    print(f"Front: {front_path}")
    print(f"Side:  {side_path}")
    print(f"Top:   {top_path}")
    print(f"Slices: {num_slices}")
    print("="*70)

    experiments = [
        ("E1", "equal", "Equal weights (0.5, 0.5)"),
        ("E2", "front_heavy", "Front-heavy (0.6, 0.4)"),
        ("E3", "side_heavy", "Side-heavy (0.4, 0.6)"),
        ("E4", "adaptive", "Adaptive weights based on confidence"),
    ]

    all_results = []

    for exp_name, fusion_strategy, description in experiments:
        print(f"\n{exp_name}: {description}")
        result = run_experiment(
            exp_name,
            fusion_strategy,
            front_path,
            side_path,
            top_path,
            num_slices
        )
        if result:
            all_results.append(result)

    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"{'Experiment':<12} {'Front IoU':<12} {'Side IoU':<12} {'Top IoU':<12} {'Avg IoU':<12}")
    print("-"*70)

    for result in all_results:
        exp_name = result['strategy_name']
        front_iou = result['iou_scores'].get('front', 'N/A')
        side_iou = result['iou_scores'].get('side', 'N/A')
        top_iou = result['iou_scores'].get('top', 'N/A')
        avg_iou = result.get('average_iou', 'N/A')

        # Format values
        front_str = f"{front_iou:.4f}" if isinstance(front_iou, float) else str(front_iou)
        side_str = f"{side_iou:.4f}" if isinstance(side_iou, float) else str(side_iou)
        top_str = f"{top_iou:.4f}" if isinstance(top_iou, float) else str(top_iou)
        avg_str = f"{avg_iou:.4f}" if isinstance(avg_iou, float) else str(avg_iou)

        print(f"{exp_name:<12} {front_str:<12} {side_str:<12} {top_str:<12} {avg_str:<12}")

    print("="*70)

    # Save results to JSON
    output_file = Path(__file__).parent / "exp_e_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    # Find best strategy
    best_avg = max(all_results, key=lambda r: r.get('average_iou', 0) or 0)
    print(f"\nBest Average IoU: {best_avg['strategy_name']} ({best_avg.get('average_iou', 0):.4f})")

    # Check if front/side improved
    baseline_front_side = 0.65  # Baseline is 0.64-0.66
    for result in all_results:
        front_iou = result['iou_scores'].get('front')
        side_iou = result['iou_scores'].get('side')

        if front_iou and side_iou:
            avg_front_side = (front_iou + side_iou) / 2
            improvement = avg_front_side - baseline_front_side
            print(f"{result['strategy_name']}: Front/Side avg = {avg_front_side:.4f} (Δ{improvement:+.4f})")

    print("="*70)

    return all_results


if __name__ == "__main__":
    # Default test images (can be overridden via command line)
    import sys

    if len(sys.argv) >= 4:
        front_path = sys.argv[1] if sys.argv[1] != "None" else None
        side_path = sys.argv[2] if sys.argv[2] != "None" else None
        top_path = sys.argv[3] if sys.argv[3] != "None" else None
    else:
        # Use default test images
        test_images_dir = Path(__file__).parent / "test_images"
        front_path = str(test_images_dir / "simple_front.png")
        side_path = str(test_images_dir / "simple_side.png")
        top_path = str(test_images_dir / "simple_top.png")

    # Run all experiments
    results = run_all_experiments(front_path, side_path, top_path, num_slices=12)

    print("\nExperiments complete!")
