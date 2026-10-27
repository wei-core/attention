# Measurement conditions

Exploratory CUDA Event timings on RTX 5060, CUDA toolkit 12.8, PyTorch 2.9.1+cu128, Python 3.12. Each shape uses 20 warmups and 30 timed calls, BF16, Hq=32, Hkv=8, D=128, block size 16. Split timings include internal scratch allocation and both launches. Another GPU process was active during measurements, so these numbers are not an isolated performance baseline and should not be used to claim a speedup. Re-run on an idle device and record clocks, power state, driver, source revision and dispersion before comparisons.

These measurements were taken before the subsequent block-reduction synchronization fix; remeasure the corrected kernel before using the results as a baseline.
