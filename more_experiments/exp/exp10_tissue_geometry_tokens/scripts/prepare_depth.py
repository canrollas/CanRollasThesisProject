#!/usr/bin/env python3
"""Cache raw Depth Anything V2 relative inverse-depth predictions.

The default Hugging Face backend uses cached local weights and performs no
downloads. The optional official-repository backend needs an existing local
checkout and checkpoint. These maps are image-derived priors, not measured
physical wound depth. Cache provenance binds model weights and source images.
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = REPO_ROOT / "datasets/wound-stage-depth-anything-v2-small"
STAGES = ("stage_1", "stage_2", "stage_3", "stage_4")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=REPO_ROOT / "datasets/wound-stage-classification")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--backend", choices=("transformers", "official"), default="transformers")
    parser.add_argument("--model-id", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--allow-download", action="store_true",
                        help="Allow Hugging Face downloads; default is local cached files only")
    parser.add_argument("--model-repo", type=Path, help="Existing official Depth-Anything-V2 checkout")
    parser.add_argument("--checkpoint", type=Path, help="Existing official .pth checkpoint")
    parser.add_argument("--encoder", choices=("vits", "vitb", "vitl", "vitg"), default="vits")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default=None)
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--limit", type=int, help="Smoke subset; requires a separate --output-root")
    parser.add_argument("--verify-only", action="store_true",
                        help="Read-only verification of all recorded image/map hashes; no model load")
    args = parser.parse_args(argv)
    if args.input_size < 14:
        parser.error("--input-size must be at least 14")
    if args.limit is not None and (args.limit < 1 or args.output_root.resolve() == DEFAULT_OUTPUT.resolve()):
        parser.error("--limit must be positive and use a separate --output-root")
    if args.backend == "official" and (args.model_repo is None or args.checkpoint is None):
        parser.error("--backend official requires --model-repo and --checkpoint")
    return args


def hf_checkpoint_hashes(model_id, revision, local_files_only):
    from huggingface_hub import hf_hub_download
    local_path = Path(model_id)

    def resolve(filename):
        if local_path.is_dir():
            path = local_path / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            return path
        return Path(hf_hub_download(model_id, filename, revision=revision,
                                    local_files_only=local_files_only))

    for candidate in ("model.safetensors", "pytorch_model.bin"):
        try:
            path = resolve(candidate)
            return {candidate: sha256(path)}
        except (FileNotFoundError, OSError):
            pass
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        try:
            index_path = resolve(index_name)
            filenames = set(json.loads(index_path.read_text())["weight_map"].values())
            return {name: sha256(resolve(name)) for name in sorted(filenames)}
        except (FileNotFoundError, OSError):
            pass
    raise FileNotFoundError("Cannot locate loaded model weight files to record checkpoint SHA256")


def build_predictor(args, device):
    import numpy as np
    import torch
    import torch.nn.functional as functional

    if args.backend == "transformers":
        import transformers
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        options = {"revision": args.revision, "local_files_only": not args.allow_download}
        processor = AutoImageProcessor.from_pretrained(args.model_id, **options)
        model = AutoModelForDepthEstimation.from_pretrained(args.model_id, **options).to(device).eval()

        @torch.inference_mode()
        def predict(image):
            inputs = processor(images=image, return_tensors="pt",
                               size={"height": args.input_size, "width": args.input_size})
            inputs = {key: value.to(device) for key, value in inputs.items()}
            depth = model(**inputs).predicted_depth
            resized = functional.interpolate(depth.unsqueeze(1), size=(image.height, image.width),
                                             mode="bicubic", align_corners=False)
            return resized[0, 0].cpu().numpy().astype(np.float32)

        metadata = {
            "model_id": args.model_id, "revision": args.revision,
            "resolved_commit": getattr(model.config, "_commit_hash", None),
            "checkpoint_sha256": hf_checkpoint_hashes(args.model_id, args.revision,
                                                       not args.allow_download),
            "processor": processor.to_dict(), "transformers_version": transformers.__version__,
            "output_resize": "bicubic, align_corners=False",
        }
        return predict, metadata

    model_repo = args.model_repo.resolve()
    source = model_repo / "depth_anything_v2/dpt.py"
    if not source.is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError("Official backend needs local depth_anything_v2/dpt.py and checkpoint")
    sys.path.insert(0, str(model_repo))
    from depth_anything_v2.dpt import DepthAnythingV2
    configs = {
        "vits": {"features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }
    model = DepthAnythingV2(encoder=args.encoder, **configs[args.encoder])
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model = model.to(device).eval()

    @torch.inference_mode()
    def predict(image):
        # Official infer_image takes OpenCV BGR, whereas PIL delivers RGB.
        bgr = np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
        return np.asarray(model.infer_image(bgr, args.input_size), dtype=np.float32)

    return predict, {
        "model_repo": str(model_repo), "encoder": args.encoder,
        "checkpoint_sha256": {str(args.checkpoint.resolve()): sha256(args.checkpoint)},
        "source_sha256": sha256(source), "output_resize": "official infer_image",
    }


def verify_existing_cache(args, files, relative_paths):
    """Audit a completed cache using its recorded provenance, without rewriting it."""
    import numpy as np
    from PIL import Image
    from data import validate_depth

    provenance_path = args.output_root / "provenance.json"
    manifest_path = args.output_root / "manifest.json"
    provenance = json.loads(provenance_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    config = provenance["config"]
    # The recorded absolute source directory is informational. A copied cache
    # may move to Colab/another checkout; the file list and byte hashes below
    # establish identity without depending on the machine's directory layout.
    if config["source_files"] != relative_paths or set(manifest) != set(relative_paths):
        raise ValueError("Source list or completed manifest differs from requested images")
    for image_path, relative_path in zip(files, relative_paths):
        row = manifest[relative_path]
        expected_output = Path(relative_path).with_suffix(".npy").as_posix()
        if row["output_path"] != expected_output:
            raise ValueError(f"Unexpected map path in manifest: {relative_path}")
        depth_path = args.output_root / expected_output
        if row["source_sha256"] != sha256(image_path) or row["depth_sha256"] != sha256(depth_path):
            raise ValueError(f"Source or relative-depth hash mismatch: {relative_path}")
        with Image.open(image_path) as image:
            expected_shape = (image.height, image.width)
        depth = np.load(depth_path, allow_pickle=False)
        validate_depth(depth, expected_shape, depth_path)
        if row["shape"] != list(depth.shape) or row["dtype"] != str(depth.dtype):
            raise ValueError(f"Recorded map format differs: {relative_path}")
    print(f"Verified {len(files)} source images and relative-depth maps; all hashes, shapes, "
          f"types and finite/nonflat checks passed. Cache unchanged: {args.output_root}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    # Heavy imports deliberately follow argument parsing so --help is standalone.
    import numpy as np
    import torch
    from PIL import Image
    from data import validate_depth

    files = []
    for stage in STAGES:
        folder = args.data_root / stage
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing staging image directory: {folder}")
        files.extend(sorted(path for path in folder.iterdir()
                            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS))
    if not files:
        raise ValueError("No staging images found")
    if args.limit is not None:
        files = files[:args.limit]
    relative_paths = [path.relative_to(args.data_root).as_posix() for path in files]
    outputs = [str(Path(path).with_suffix(".npy")) for path in relative_paths]
    if len(set(outputs)) != len(outputs):
        raise ValueError("Duplicate image stems would collide in the relative-depth cache")
    if args.verify_only:
        verify_existing_cache(args, files, relative_paths)
        return
    device = args.device or ("cuda" if torch.cuda.is_available() else
                            "mps" if torch.backends.mps.is_available() else "cpu")
    started = time.monotonic()
    predict, model_metadata = build_predictor(args, device)
    config = {
        "schema_version": 1, "backend": args.backend, "input_size": args.input_size,
        "data_root": str(args.data_root.resolve()), "device": device,
        "torch_version": torch.__version__, "model": model_metadata,
        "source_files": relative_paths, "limit": args.limit,
        "output": "raw float32 relative inverse depth at original RGB dimensions",
        "interpretation": "image-derived geometric prior; no metric or measured wound depth",
        "generator_sha256": sha256(Path(__file__)),
    }
    # Processor configurations contain tuples; compare their persisted JSON
    # representation so a resumed run does not confuse tuples with lists.
    config = json.loads(json.dumps(config))
    args.output_root.mkdir(parents=True, exist_ok=True)
    provenance_path = args.output_root / "provenance.json"
    manifest_path = args.output_root / "manifest.json"
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        if provenance.get("config") != config:
            previous = provenance.get("config", {})
            changed = {key for key in set(previous) | set(config) if previous.get(key) != config.get(key)}
            completed = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
            if changed == {"generator_sha256"} and set(completed) == set(relative_paths):
                # Reusing a COMPLETE, verified cache does not execute a new
                # inference implementation. Preserve its original provenance;
                # never mix old and new generators in an unfinished cache.
                verify_existing_cache(args, files, relative_paths)
                print("Complete existing cache retained with its recorded generator; 0 maps generated", flush=True)
                return
            raise ValueError("Output provenance differs from this run; choose a fresh --output-root")
    else:
        if any(args.output_root.iterdir()):
            raise ValueError("Refusing to adopt a nonempty cache without provenance.json")
        write_json(provenance_path, {
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": config,
        })
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    count_new = 0
    for index, (image_path, relative_path, output_rel) in enumerate(zip(files, relative_paths, outputs), 1):
        output_path = args.output_root / output_rel
        image_hash = sha256(image_path)
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
        if output_path.exists():
            row = manifest.get(relative_path)
            if row is None or row["source_sha256"] != image_hash or row["depth_sha256"] != sha256(output_path):
                raise ValueError(f"Existing cache content/provenance mismatch: {output_path}")
            validate_depth(np.load(output_path, allow_pickle=False), (image.height, image.width), output_path)
            print(f"[{index}/{len(files)}] verified existing {relative_path}", flush=True)
            continue
        if relative_path in manifest:
            raise ValueError(f"Manifest points to a missing output: {output_path}")
        inference_start = time.monotonic()
        depth = predict(image)
        validate_depth(depth, (image.height, image.width), output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(".npy.tmp")
        with open(temporary, "wb") as output_stream:
            np.save(output_stream, depth, allow_pickle=False)
        temporary.replace(output_path)
        manifest[relative_path] = {
            "source_sha256": image_hash, "depth_sha256": sha256(output_path),
            "output_path": output_rel, "shape": list(depth.shape), "dtype": str(depth.dtype),
            "min": float(depth.min()), "max": float(depth.max()), "std": float(depth.std()),
        }
        write_json(manifest_path, manifest)
        count_new += 1
        print(f"[{index}/{len(files)}] {relative_path} {time.monotonic()-inference_start:.2f}s "
              f"range=({depth.min():.4f},{depth.max():.4f})", flush=True)
    print(f"Complete: {len(files)} maps verified, {count_new} generated; "
          f"elapsed {time.monotonic()-started:.2f}s; {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
