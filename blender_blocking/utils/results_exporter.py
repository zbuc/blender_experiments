"""Export E2E test results for GitHub Pages visualization."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class E2EResultsExporter:
    """Export E2E test results to a web-friendly format."""

    def __init__(self, output_dir: Path):
        """
        Initialize exporter.

        Args:
            output_dir: Directory to export results to (e.g., docs/e2e-results)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(exist_ok=True)

    def export_test_case(
        self,
        test_name: str,
        views: List[str],
        reference_paths: Dict[str, str],
        rendered_paths: Dict[str, str],
        results: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Export a single test case.

        Args:
            test_name: Name of the test case (e.g., "vase_legacy_default")
            views: List of views tested (e.g., ["front", "side", "top"])
            reference_paths: Dict mapping view names to reference image paths
            rendered_paths: Dict mapping view names to rendered image paths
            results: Dict containing IoU and other metrics per view
            metadata: Optional metadata (config, timestamp, etc.)
        """
        # Copy images to output directory
        test_images_dir = self.images_dir / test_name
        test_images_dir.mkdir(exist_ok=True)

        image_info = {}
        for view in views:
            if view in reference_paths and view in rendered_paths:
                ref_path = Path(reference_paths[view])
                render_path = Path(rendered_paths[view])

                # Copy reference image
                ref_dest = test_images_dir / f"{view}_reference.png"
                if ref_path.exists():
                    shutil.copy2(ref_path, ref_dest)

                # Copy rendered image
                render_dest = test_images_dir / f"{view}_rendered.png"
                if render_path.exists():
                    shutil.copy2(render_path, render_dest)

                image_info[view] = {
                    "reference": f"images/{test_name}/{view}_reference.png",
                    "rendered": f"images/{test_name}/{view}_rendered.png",
                }

                # Copy debug images if they exist
                debug_dir = ref_path.parent.parent / "debug_silhouettes"
                for suffix in ["ref_canon", "render_canon", "diff"]:
                    debug_path = debug_dir / f"{view}_{suffix}.png"
                    if debug_path.exists():
                        debug_dest = test_images_dir / f"{view}_{suffix}.png"
                        shutil.copy2(debug_path, debug_dest)
                        if "debug" not in image_info[view]:
                            image_info[view]["debug"] = {}
                        image_info[view]["debug"][suffix] = (
                            f"images/{test_name}/{view}_{suffix}.png"
                        )

        # Calculate summary stats
        ious = [r["iou"] for r in results.values() if "iou" in r]
        avg_iou = sum(ious) / len(ious) if ious else 0.0
        min_iou = min(ious) if ious else 0.0
        max_iou = max(ious) if ious else 0.0

        # Create test case JSON
        test_case_data = {
            "test_name": test_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "summary": {
                "avg_iou": round(avg_iou, 4),
                "min_iou": round(min_iou, 4),
                "max_iou": round(max_iou, 4),
                "num_views": len(views),
                "passed": avg_iou >= 0.7,  # Default threshold
            },
            "views": {},
            "metadata": metadata or {},
        }

        for view in views:
            if view in results:
                test_case_data["views"][view] = {
                    "iou": round(results[view]["iou"], 4),
                    "intersection": results[view].get("intersection", 0),
                    "union": results[view].get("union", 0),
                    "pixel_difference": round(
                        results[view].get("pixel_difference", 0.0), 4
                    ),
                    "images": image_info.get(view, {}),
                }

        # Save test case JSON
        test_case_file = self.output_dir / f"{test_name}.json"
        with open(test_case_file, "w") as f:
            json.dump(test_case_data, f, indent=2)

        print(f"✓ Exported test case: {test_case_file}")

    def export_index(self, test_cases: List[str]) -> None:
        """
        Export an index.json listing all test cases.

        Args:
            test_cases: List of test case names
        """
        index_data = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "test_cases": test_cases,
        }

        index_file = self.output_dir / "index.json"
        with open(index_file, "w") as f:
            json.dump(index_data, f, indent=2)

        print(f"✓ Exported index: {index_file}")
