# ViT benchmarks

from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch
from datasets import load_dataset
import neuronperf.torch
import torch_neuron
import os


pipeline_sizes = [1, 4]
batch_sizes = [1, 2, 3, 4]

for ncp in pipeline_sizes:
    for bs in batch_sizes:
        model_file = f"vit_neuron_ncp{ncp}_bs{bs}.pt"
        report_file = f"vit_neuron_ncp{ncp}_bs{bs}_benchmark.csv"
        inputs = torch.randn(bs, 3, 224, 224)
        print(f"ncp: {ncp}  bs: {bs}")

        if not os.path.exists(report_file):
            reports = neuronperf.torch.benchmark(model_filename=model_file, inputs=inputs, batch_sizes=[bs], pipeline_sizes=[ncp])
            neuronperf.print_reports(reports)
            neuronperf.write_csv(reports, report_file)
        else:
            print(f"Report file {report_file} already exists. Skipping this benchmark run.\n")
