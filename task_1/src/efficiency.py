import time

import torch


def count_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def measure_latency(model, input_size, device, batch_size=1, warmup=10, iters=50):
    model.eval()
    dummy = torch.randn(batch_size, 3, input_size, input_size, device=device)

    for _ in range(warmup):
        model(dummy)

    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            model(dummy)
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            model(dummy)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return elapsed_ms / iters
