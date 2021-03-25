# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from tensorflow_serving.apis import predict_pb2 as tensorflow__serving_dot_apis_dot_predict__pb2


class PredictionServiceStub(object):
    """open source marker; do not remove
    PredictionService provides access to machine-learned models loaded by
    model_servers.
    Classify.
    rpc Classify(ClassificationRequest) returns (ClassificationResponse);
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Predict = channel.unary_unary(
                '/tensorflow.serving.PredictionService/Predict',
                request_serializer=tensorflow__serving_dot_apis_dot_predict__pb2.PredictRequest.SerializeToString,
                response_deserializer=tensorflow__serving_dot_apis_dot_predict__pb2.PredictResponse.FromString,
                )


class PredictionServiceServicer(object):
    """open source marker; do not remove
    PredictionService provides access to machine-learned models loaded by
    model_servers.
    Classify.
    rpc Classify(ClassificationRequest) returns (ClassificationResponse);
    """

    def Predict(self, request, context):
        """Regress.
        rpc Regress(RegressionRequest) returns (RegressionResponse);

        Predict -- provides access to loaded TensorFlow model.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_PredictionServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Predict': grpc.unary_unary_rpc_method_handler(
                    servicer.Predict,
                    request_deserializer=tensorflow__serving_dot_apis_dot_predict__pb2.PredictRequest.FromString,
                    response_serializer=tensorflow__serving_dot_apis_dot_predict__pb2.PredictResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'tensorflow.serving.PredictionService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class PredictionService(object):
    """open source marker; do not remove
    PredictionService provides access to machine-learned models loaded by
    model_servers.
    Classify.
    rpc Classify(ClassificationRequest) returns (ClassificationResponse);
    """

    @staticmethod
    def Predict(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/tensorflow.serving.PredictionService/Predict',
            tensorflow__serving_dot_apis_dot_predict__pb2.PredictRequest.SerializeToString,
            tensorflow__serving_dot_apis_dot_predict__pb2.PredictResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)