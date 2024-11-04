import onnx
import numpy as np
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

def compare_models(model_path1, model_path2):
    # Load ONNX models
    model1 = onnx.load(model_path1)
    model2 = onnx.load(model_path2)
    
    # Compare graph structures
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

# Paths to ONNX models
model_path1 = "model1.onnx"
model_path2 = "model2.onnx"

# Compare models
compare_models(model_path1, model_path2)
