"""
export_model.py — export the fine-tuned PyTorch weights to ONNX (required by
Part 2's C++ pipeline) and, as a bonus, to OpenVINO IR.

The ONNX export is deliberately static-shaped and has NMS baked out
(nms=False): the C++ side wants a fixed [1,3,416,416] input and performs its
own thresholding + NMS on the raw [1, 4+nc, num_anchors] output tensor.

Usage:
    python export_model.py --weights models/best.pt --imgsz 416 --formats onnx,openvino
"""
import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default="models/best.pt")
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--opset", type=int, default=12)
    p.add_argument("--formats", default="onnx,openvino", help="Comma-separated: onnx,openvino")
    return p.parse_args()


def export_onnx(weights: str, imgsz: int, opset: int) -> Path:
    model = YOLO(weights)
    try:
        path = model.export(format="onnx", imgsz=imgsz, opset=opset, dynamic=False,
                             simplify=True, nms=False, device="cpu")
    except Exception as e:
        print(f"simplify=True failed ({e}); retrying without simplification")
        path = model.export(format="onnx", imgsz=imgsz, opset=opset, dynamic=False,
                             simplify=False, nms=False, device="cpu")
    return Path(path)


def verify_onnx(onnx_path: Path, imgsz: int, num_classes: int) -> None:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]
    print(f"ONNX input : {inp.name} {inp.shape}")
    print(f"ONNX output: {out.name} {out.shape}")

    dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
    result = session.run(None, {inp.name: dummy})[0]
    expected_channels = 4 + num_classes
    assert result.shape[1] == expected_channels, (
        f"expected output channel dim {expected_channels} (4 box + {num_classes} classes), "
        f"got {result.shape}"
    )
    print(f"Verified: output shape {result.shape} matches 4 + {num_classes} classes. "
          f"(This is the exact layout src/detector.cpp in Part 2 is written against.)")


def export_openvino(weights: str, imgsz: int) -> Path | None:
    try:
        import openvino  # noqa: F401
    except ImportError:
        print("openvino not installed -- skipping OpenVINO export "
              "(pip install openvino to enable this bonus format)")
        return None
    model = YOLO(weights)
    path = model.export(format="openvino", imgsz=imgsz, dynamic=False, device="cpu")
    return Path(path)


def main() -> None:
    args = parse_args()
    formats = {f.strip().lower() for f in args.formats.split(",")}

    model = YOLO(args.weights)
    num_classes = len(model.names)
    print(f"Loaded {args.weights} with classes: {model.names}")

    if "onnx" in formats:
        onnx_path = export_onnx(args.weights, args.imgsz, args.opset)
        print(f"Exported ONNX -> {onnx_path}")
        verify_onnx(onnx_path, args.imgsz, num_classes)

    if "openvino" in formats:
        ov_path = export_openvino(args.weights, args.imgsz)
        if ov_path:
            print(f"Exported OpenVINO -> {ov_path}")


if __name__ == "__main__":
    main()
