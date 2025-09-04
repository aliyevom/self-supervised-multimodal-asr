import os
import sys
from pathlib import Path

def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    protos_dir = repo_root / "protos"
    out_dir = repo_root / "src"

    # Ensure import paths
    sys.path.insert(0, str(repo_root))

    try:
        from grpc_tools import protoc  # type: ignore
    except Exception as exc:  # pragma: no cover
        print("grpcio-tools is required. Install with: pip install grpcio-tools", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 1

    proto_files = [
        str(p.relative_to(repo_root))
        for p in protos_dir.rglob("*.proto")
    ]

    if not proto_files:
        print("No .proto files found", file=sys.stderr)
        return 1

    args = [
        "protoc",
        f"-I{protos_dir}",
        f"-I{repo_root}",
        f"--python_out={out_dir}",
        f"--grpc_python_out={out_dir}",
    ] + proto_files

    result = protoc.main(args)
    if result != 0:
        print(f"protoc failed with exit code {result}", file=sys.stderr)
        return result

    # Touch package __init__ files to ensure packages are importable
    asr_pkg = out_dir / "asr" / "v1"
    asr_pkg.mkdir(parents=True, exist_ok=True)
    (out_dir / "asr").mkdir(parents=True, exist_ok=True)
    (out_dir / "asr" / "__init__.py").touch()
    (asr_pkg / "__init__.py").touch()
    print("Generated protobuf stubs into 'src' successfully.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())


