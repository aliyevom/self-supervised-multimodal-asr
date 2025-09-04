from __future__ import annotations

from typing import Iterable
import grpc

from asr.v1 import context_service_pb2 as ctx_pb2
from asr.v1 import context_service_pb2_grpc as ctx_grpc
from asr.v1 import common_pb2


class ContextEncoderServiceImpl(ctx_grpc.ContextEncoderServiceServicer):
    pass

    def StreamContext(
        self, request_iterator: Iterable[ctx_pb2.ContextStreamRequest], context: grpc.ServicerContext
    ) -> Iterable[ctx_pb2.ContextStreamResponse]:
        dim = 512
        for req in request_iterator:
            if req.WhichOneof("payload") == "config":
                if req.config and req.config.extra.get("context_dim"):
                    try:
                        dim = int(req.config.extra["context_dim"].number_value)
                    except Exception:
                        pass
        yield ctx_pb2.ContextStreamResponse(
            vector=common_pb2.ContextVector(values=[0.0] * dim, dim=dim)
        )

    def EncodeContext(self, request: ctx_pb2.EncodeContextRequest, context: grpc.ServicerContext) -> ctx_pb2.EncodeContextResponse:
        dim = 512
        return ctx_pb2.EncodeContextResponse(
            status=common_pb2.Status(code=common_pb2.Status.OK, message="ok"),
            vector=common_pb2.ContextVector(values=[0.0] * dim, dim=dim),
        )


