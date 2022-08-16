from transformers import BeitFeatureExtractor, BeitForImageClassification
import torch
from datasets import load_dataset
import neuronperf.torch
import torch_neuron
import os

feature_extractor = BeitFeatureExtractor.from_pretrained("microsoft/beit-base-patch16-224")
model = BeitForImageClassification.from_pretrained("microsoft/beit-base-patch16-224")
model = model.eval()

pipeline_sizes = [1, 4]
batch_sizes = [1, 2, 3, 4, 5, 6]

for ncp in pipeline_sizes:
    for bs in batch_sizes:
        model_file = f"beit_neuron_ncp{ncp}_bs{bs}.pt"
        inputs = torch.randn(bs, 3, 224, 224)
        print(f"ncp: {ncp}  bs: {bs}")

        if not os.path.exists(model_file):
            print("Attempting model compilation")
            nmod = torch.neuron.trace(model, example_inputs=inputs, compiler_args=['--neuroncore-pipeline-cores', f"{ncp}"], strict=False)
            nmod.save(model_file)
            del(nmod) # we need to release the model from memory so it doesn't affect benchmarking later on
        else:
            print(f"Found previously compiled model. Skipping compilation\n")
    
