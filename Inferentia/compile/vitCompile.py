from transformers import ViTFeatureExtractor, ViTForImageClassification
from datasets import load_dataset
import torch
import neuronperf.torch
import torch_neuron
import os

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
model = model.eval()

pipeline_sizes = [1, 4]
batch_sizes = [1, 2]

for ncp in pipeline_sizes:
    for bs in batch_sizes:
        model_file = f"vit_neuron_ncp{ncp}_bs{bs}.pt"
        inputs = torch.randn(bs, 3, 224, 224)
        print(f"ncp: {ncp}  bs: {bs}")

        if not os.path.exists(model_file):
            print("Attempting model compilation")
            nmod = torch.neuron.trace(model, example_inputs=inputs, compiler_args=['--neuroncore-pipeline-cores', f"{ncp}"], strict=False)
            nmod.save(model_file)
            del(nmod) # we need to release the model from memory so it doesn't affect benchmarking later on
        else:
            print(f"Found previously compiled model. Skipping compilation\n")
    
