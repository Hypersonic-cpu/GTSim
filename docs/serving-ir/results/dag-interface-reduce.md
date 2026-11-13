# DAG interface reduction benchmark

Frozen baseline: `fc1226489474b452430d1b1b53a1aa2465adde8b`; each run had a 900 s hard timeout. CPU is total user+system time and RSS is MiB. `Combined RSS` is the sampled simultaneous Python+C++ peak.

| Case | Version | Cycles | Python CPU | C++ CPU | Python RSS | C++ RSS | Combined RSS | Wall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| T1 | Old FIFO | 38948 | 1.01 | 1.61 | 29.21 | 25.73 | 54.94 | 2.663 |
| T1 | Previous Socket | 38948 | 1.00 | 1.84 | 24.32 | 40.15 | 63.77 | 2.762 |
| T1 | Refactored Socket | 38948 | 1.02 | 1.60 | 24.53 | 41.04 | 65.57 | 2.698 |
| T2 | Old FIFO | 14914 | 1.01 | 0.79 | 29.05 | 36.15 | 65.20 | 1.907 |
| T2 | Previous Socket | 14914 | 1.00 | 0.97 | 24.20 | 40.15 | 63.51 | 1.898 |
| T2 | Refactored Socket | 14914 | 1.02 | 0.80 | 24.60 | 41.29 | 65.83 | 1.885 |
| T3 | Old FIFO | 11769 | 0.33 | 0.43 | 23.43 | 15.82 | 39.25 | 0.874 |
| T3 | Previous Socket | 11769 | 0.34 | 0.48 | 22.65 | 37.28 | 59.93 | 0.923 |
| T3 | Refactored Socket | 11769 | 0.32 | 0.46 | 26.61 | 37.99 | 60.44 | 0.892 |
| T4 | Old FIFO | 25964 | 2.05 | 1.98 | 36.16 | 68.42 | 100.33 | 4.079 |
| T4 | Previous Socket | 25964 | 2.10 | 1.97 | 28.84 | 67.96 | 96.80 | 3.787 |
| T4 | Refactored Socket | 25964 | 2.16 | 1.58 | 28.63 | 62.69 | 91.32 | 3.775 |
| T5 | Old FIFO | 397765 | 152.97 | 45.10 | 589.89 | 839.41 | 1160.81 | 152.388 |
| T5 | Previous Socket | 397765 | 174.76 | 60.93 | 471.28 | 757.34 | 1006.47 | 160.585 |
| T5 | Refactored Socket | 397765 | 160.47 | 47.25 | 470.09 | 688.49 | 999.38 | 154.962 |

T2 decomposes to 6,824 + 8,090. The prefetch regression covers multiple CTAs and multiple pipeline iterations and verifies each eligible load receives one window consumer; all 24 frontend tests pass.

Production LOC (C++ `src/include/workloads` plus Python `frontend/src` and `frontend/tools`) fell from 22,985 to 20,139: 758 lines added, 3,604 removed, net −2,846. The C++ core/workload breakdown is 12,586→11,342 and 1,853→1,710; frontend source/tools are 5,488→5,167 and 3,058→1,920.

Final checks: 19 targeted CTest cases and all 24 frontend unit tests pass. The profile checker still reports the same four workload-builder lines as the frozen baseline; no new violation was introduced.
