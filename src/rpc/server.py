from __future__ import annotations

import grpc
from concurrent import futures
import os

from asr.v1 import asr_service_pb2_grpc as asr_grpc
from asr.v1 import context_service_pb2_grpc as ctx_grpc
from asr.v1 import admin_service_pb2_grpc as admin_grpc

from .asr_service_impl import ASRServiceImpl
from .context_service_impl import ContextEncoderServiceImpl
from .admin_service_impl import AdminServiceImpl


def serve(host: str = "0.0.0.0", port: int = 50051) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    asr_grpc.add_ASRServiceServicer_to_server(ASRServiceImpl(), server)
    ctx_grpc.add_ContextEncoderServiceServicer_to_server(ContextEncoderServiceImpl(), server)
    admin_grpc.add_AdminServiceServicer_to_server(AdminServiceImpl(), server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    print(f"ASR RPC server listening on {host}:{port}")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()


