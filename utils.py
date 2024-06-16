from nnsight import pydantics

def get_request(tracer):
    return pydantics.RequestModel(
        kwargs=tracer._kwargs,
        repo_id=tracer._model._model_key,
        batched_input=tracer._batched_input,
        intervention_graph=tracer._graph.nodes,
    )

