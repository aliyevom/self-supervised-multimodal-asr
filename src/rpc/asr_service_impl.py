from __future__ import annotations

import time
from typing import Iterable

import grpc

from asr.v1 import asr_service_pb2 as asr_pb2
from asr.v1 import asr_service_pb2_grpc as asr_grpc
from asr.v1 import common_pb2


class ASRServiceImpl(asr_grpc.ASRServiceServicer):
    pass

    def StreamingRecognize(
        self, request_iterator: Iterable[asr_pb2.StreamingRecognizeRequest], context: grpc.ServicerContext
    ) -> Iterable[asr_pb2.StreamingRecognizeResponse]:
        start = time.time()
        audio_bytes = 0
        config: common_pb2.RuntimeConfig | None = None

        for req in request_iterator:
            which = req.WhichOneof("payload")
            if which == "config":
                config = req.config
            elif which == "speech":
                audio_bytes += len(req.speech.data)
                yield asr_pb2.StreamingRecognizeResponse(
                    interim=asr_pb2.PartialTranscript(start_frame=0, end_frame=0, text="")
                )
            elif which == "context":
                pass
            elif which == "context_vector":
                pass

        elapsed = time.time() - start

        metrics = common_pb2.ProfilingMetrics(rtf=0.0, tokens_per_second=0, memory_bytes=0)

        final = asr_pb2.StreamingRecognizeResponse(
            final=common_pb2.Hypothesis(text="", score=0.0)
        )
        yield final
        yield asr_pb2.StreamingRecognizeResponse(metrics=metrics)
        yield asr_pb2.StreamingRecognizeResponse(
            status=common_pb2.Status(code=common_pb2.Status.OK, message=f"received {audio_bytes} bytes")
        )

    def Recognize(self, request: asr_pb2.RecognizeRequest, context: grpc.ServicerContext) -> asr_pb2.RecognizeResponse:
        return asr_pb2.RecognizeResponse(
            status=common_pb2.Status(code=common_pb2.Status.OK, message="ok"),
            hypothesis=common_pb2.Hypothesis(text="", score=0.0),
            metrics=common_pb2.ProfilingMetrics(),
        )


