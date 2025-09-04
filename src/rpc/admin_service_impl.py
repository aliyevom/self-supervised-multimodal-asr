from __future__ import annotations

import platform
import grpc

from asr.v1 import admin_service_pb2 as admin_pb2
from asr.v1 import admin_service_pb2_grpc as admin_grpc
from asr.v1 import common_pb2


class AdminServiceImpl(admin_grpc.AdminServiceServicer):
    def Health(self, request, context):
        device = "cuda" if False else "cpu"
        return admin_pb2.HealthResponse(
            status=common_pb2.Status(code=common_pb2.Status.OK, message="ok"),
            version="v1",
            device=device,
        )

    def GetConfig(self, request, context):
        return admin_pb2.GetConfigResponse()

    def UpdateConfig(self, request, context):
        return admin_pb2.UpdateConfigResponse(
            status=common_pb2.Status(code=common_pb2.Status.OK, message="updated")
        )

    def LoadCheckpoint(self, request, context):
        return common_pb2.Status(code=common_pb2.Status.OK, message="loaded")

    def GetProfiling(self, request, context):
        return admin_pb2.ProfilingSnapshot(status=common_pb2.Status(code=common_pb2.Status.OK))


