"""
Below pytest test compares two onnx models for identical structure and parameters.  
"""
import onnx
import numpy as np
import os
import pytest

# Paths to ONNX models
MODEL_PATH1 = "model1.onnx"
MODEL_PATH2 = "model2.onnx"

# Helper function to compare graph structures
def compare_graphs(graph1, graph2):
    nodes1 = graph1.node
    nodes2 = graph2.node
    
    if len(nodes1) != len(nodes2):
        print("Number of nodes in the graphs is different.")
        return False

    for i, (node1, node2) in enumerate(zip(nodes1, nodes2)):
        if node1 != node2:
            print(f"Node {i} is different:")
            print(f"Model 1 node: {node1}")
            print(f"Model 2 node: {node2}")
            return False
    print("Graph structures are identical.")
    return True

# Helper function to compare model parameters
def compare_parameters(model_path1, model_path2):
    # Load ONNX models
    model1 = onnx.load(model_path1)
    model2 = onnx.load(model_path2)
    
    # Compare graph structures
    if not compare_graphs(model1.graph, model2.graph):
        return False

    # Compare weights and parameters
    for i, tensor1 in enumerate(model1.graph.initializer):
        tensor2 = model2.graph.initializer[i]
        if tensor1.name != tensor2.name:
            print("Parameter names are different.")
            return False
        if not np.array_equal(tensor1.float_data, tensor2.float_data):
            print("Parameter values are different.")
            return False

    print("Models are identical.")
    return True


@pytest.mark.skipif(
    not (os.path.exists("model1.onnx") and os.path.exists("model2.onnx")),
    reason="ONNX models are not available"
)
def test_compare_onnx_models():
    # Load ONNX models
    model1 = onnx.load(MODEL_PATH1)
    model2 = onnx.load(MODEL_PATH2)

    # Perform comparisons
    assert compare_graphs(model1.graph, model2.graph), "Graph structures are different."
    assert compare_parameters(model1, model2), "Model parameters are different."

    print("ONNX models are identical.")

