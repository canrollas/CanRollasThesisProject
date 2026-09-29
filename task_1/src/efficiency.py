import time

import torch


def count_params_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


@torch.no_grad()
def measure_latency_ms(model, device, image_size=256, num_warmup=5, num_runs=20):
    model.eval().to(device)
    dummy = torch.randn(1, 3, image_size, image_size, device=device)

    for _ in range(num_warmup):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_runs):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start

    return (elapsed / num_runs) * 1000.0
