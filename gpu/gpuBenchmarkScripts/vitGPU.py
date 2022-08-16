import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
import time
from timing import Timer
import numpy as np
import pandas as pd

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.__version__)
model.to(device)

batch_size = [1, 3, 16, 32, 64, 128]


# pricing information: update this from ec2 - on demand pricing https://aws.amazon.com/ec2/pricing/on-demand/
# price as of August 8, 2022
g5_hourly_rate = 1.006



columns = ['model name', 'batch size','latency_p0', 'latency_p50', 'latency_p90', 'latency_p95', 'latency_p99', 'latency_p100', 'throughput average', 'throughput peak', 'cost per 1 m instances']
benchmarkDataFrame = pd.DataFrame(columns = columns)


index = 0
for batch in batch_size:
    x = torch.randn(batch, 3, 224, 224)
    x_to = x.cuda()

    latencies = []
    throughputs = []
    for u in range(1000):
        start = time.perf_counter()
        model(x_to)
        torch.cuda.synchronize(device)
        end = time.perf_counter() - start
        a = end * 1000
        throughput = batch / end 
        latencies.append(a)
        throughputs.append(throughput)
        print(u)

    latencies = np.array(latencies)
    throughputs = np.array(throughputs)

    latencies_ordered = np.sort(latencies, axis = -1)
    #throughputs_ordered = np.sort(latencies, axis = -1)

    latency_p100 = np.max(np.percentile(latencies_ordered, 100, -1))
    latency_p99 = np.max(np.percentile(latencies_ordered, 99, -1))
    latency_p95 = np.max(np.percentile(latencies_ordered, 95, -1))
    latency_p90 = np.max(np.percentile(latencies_ordered, 90, -1))
    latency_p50 = np.max(np.percentile(latencies_ordered, 50, -1))
    latency_p0 = np.max(np.percentile(latencies_ordered, 0, -1))

    throughput_average = np.average(throughputs, axis = -1)
    throughputs_peak = np.max(throughputs)

    # cost per 1m instances:
    time_for_1m_inferences_in_seconds = 1000000 / throughput_average
    in_hours = time_for_1m_inferences_in_seconds / 60 / 60
    cost = in_hours * g5_hourly_rate

    print('latency_p0: {:.6f}ms'.format(latency_p0))
    print('latency_p50: {:.6f}ms'.format(latency_p50))
    print('latency_p90: {:.6f}ms'.format(latency_p90))
    print('latency_p95: {:.6f}ms'.format(latency_p95))
    print('latency_p99: {:.6f}ms'.format(latency_p99))
    print('latency_100: {:.6f}ms'.format(latency_p100))
    print('throughput_average: {:.6f}'.format(throughput_average))
    print('throughput_peak: {:.6f}'.format(throughputs_peak))
    print('Cost per 1m instances: {:.6f}'.format(cost))



   
    benchmarkDataFrame.loc[index] = ['vit-base-patch16-224', batch, latency_p0, latency_p50, latency_p90, latency_p95, latency_p99, latency_p100, throughput_average, throughputs_peak, cost]
    index += 1



benchmarkDataFrame.to_csv('vit_gpu_benchmark_2.csv', index = False)






