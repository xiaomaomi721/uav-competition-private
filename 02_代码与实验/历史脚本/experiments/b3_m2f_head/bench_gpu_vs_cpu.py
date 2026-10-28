import numpy as np, time, torch

torch.set_num_threads(4)
HLO, HHI, HBIN = -1.5, 1.5, 1e-5
NEDGES = int(round((HHI - HLO) / HBIN)) + 1
edges = np.linspace(HLO, HHI, NEDGES)
rng = np.random.default_rng(0)
tau = rng.uniform(-1.2, 1.2, 1_000_000).astype(np.float64)
print("n =", tau.size, " bins =", NEDGES - 1)

t = time.time()
for _ in range(5):
    np.histogram(tau, bins=edges)
a = (time.time() - t) / 5
print("np.histogram(edges)        %8.2f ms/call" % (a * 1000))

t = time.time()
for _ in range(5):
    idx = ((tau - HLO) / HBIN).astype(np.int64)
    np.clip(idx, 0, NEDGES - 2, out=idx)
    np.bincount(idx, minlength=NEDGES - 1)
b = (time.time() - t) / 5
print("arithmetic idx + bincount  %8.2f ms/call" % (b * 1000))
print("speedup %.1fx" % (a / b))

h1, _ = np.histogram(tau, bins=edges)
idx = np.clip(((tau - HLO) / HBIN).astype(np.int64), 0, NEDGES - 2)
h2 = np.bincount(idx, minlength=NEDGES - 1)
print("identical:", bool((h1 == h2).all()), " max diff:", int(np.abs(h1 - h2).max()))

# GPU2 vs CPU on the elementwise/reduction part this workload is made of
n = 1_000_000
dev = 'cuda:0'
s = torch.rand(n, 8, device=dev)
bb = torch.zeros(8, device=dev)
torch.cuda.synchronize()
t = time.time()
for _ in range(5):
    (s + bb).argmax(-1)
torch.cuda.synchronize()
print("GPU2 argmax(1M x 8)        %8.2f ms" % ((time.time() - t) / 5 * 1000))

s2 = torch.rand(n, 8)
bb2 = torch.zeros(8)
t = time.time()
for _ in range(5):
    (s2 + bb2).argmax(-1)
print("CPU  argmax(1M x 8)        %8.2f ms" % ((time.time() - t) / 5 * 1000))

free, total = torch.cuda.mem_get_info()
print("GPU2 %s  free %.1f / %.1f GiB" % (torch.cuda.get_device_name(0), free / 2**30, total / 2**30))

# end-to-end transfer cost: how long to move one 1024x1024x8 fp32 map to GPU2
x = torch.rand(1024, 1024, 8)
torch.cuda.synchronize()
t = time.time()
for _ in range(20):
    y = x.to(dev)
    torch.cuda.synchronize()
print("H2D 1024x1024x8 fp32       %8.2f ms  (%.1f MiB)" % ((time.time() - t) / 20 * 1000, x.numel() * 4 / 2**20))
