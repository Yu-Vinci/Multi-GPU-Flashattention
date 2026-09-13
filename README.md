# Multi-GPU FlashAttention — Ring Attention Research Prototype

Research code for attention workload modeling, static scheduling, and Ring Attention execution with CUDA and MPI/NCCL.

This repository contains the earlier C/CUDA prototype. Scheduling experiments and GPU execution are separate components in this published snapshot. The newer local C++ revision and its experiment results are not included here.

## Where to start

| Area | Purpose | Guide |
| --- | --- | --- |
| Workload and scheduling | Task generation, Round-Robin, LPT and contiguous partitions | [workload/README.md](workload/README.md) |
| Ring Attention | Benchmark variants, reference checking and shard debugging | [ring/attention/README.md](ring/attention/README.md) |
| Communication only | Ring data exchange without attention | [ring/loop/README.md](ring/loop/README.md) |
| Single-GPU experiments | Python timing scripts using the external `flash-attn` package | [tests/](tests/) |

The related [ring-attention-benchmark](https://github.com/3429495086/ring-attention-benchmark) repository packages communication and attention experiments with a shared Makefile and cluster launch scripts. Use this repository to explore the research components, and that package for its standalone benchmark workflow.

## Published implementation

- C workload generation and block/row scheduling baselines.
- Staged MPI, CUDA-aware MPI and NCCL ring experiments.
- Blocking and nonblocking MPI variants, with staged/CUDA-aware overlap variants.
- Output dumps, a CPU full-reference checker and shard-owner debug output.

The local CUDA attention kernels are research baselines. This repository is not the upstream FlashAttention library or a full-model training implementation. Python experiments call `flash-attn` separately; comparisons with the ring benchmarks must match shapes, precision and timing scope.

## Quick start

### CPU workload demo

Requires GCC or Clang with C11 support; no GPU is needed.

```bash
git clone https://github.com/3429495086/Multi-GPU-Flashattention.git
cd Multi-GPU-Flashattention/workload
gcc -O2 -std=c11 -o demo demo.c scheduler.c workload.c -lm
./demo --seq 4096 --gpus 2 --mask causal
```

### GPU correctness check

From the cloned repository root, run on a GPU server with a CUDA toolkit, Python 3 and CUDA-aware MPI. Replace the installation paths before running. NCCL is needed for the separate NCCL benchmarks.

```bash
export CUDA_HOME=/path/to/cuda
export MPI_HOME=/path/to/cuda-aware-mpi
cd ring/attention/bench
NP=2 SIZE=262144 WARMUP=1 ITERS=1 bash check_attention_correctness.sh
```

The script builds four MPI attention variants and the CPU verifier, checks each output against the full reference, and compares nonblocking baselines with their overlap variants. A passing run covers the tested configuration, not every possible input or GPU topology.

See [the attention guide](ring/attention/README.md) for build details, output formats and `ATTENTION_PRINT_SHARDS=1` debugging.

## Repository layout

```text
.
├── workload/             # C workload model and scheduling experiments
├── ring/
│   ├── loop/             # Communication-only experiments
│   └── attention/
│       ├── bench/        # Benchmark variants and correctness script
│       ├── common/       # Device and shard helpers
│       ├── gpu/          # Earlier GPU variants
│       └── verify_full.c # CPU reference
└── tests/                # Python single-GPU timing scripts
```

The Python scripts require PyTorch with CUDA support and `flash-attn`. They are separate from the MPI benchmark build.

## Scope and next steps

This published snapshot explores ring communication, overlap and correctness. Runtime integration of scheduling decisions, communication-aware scheduling and heterogeneous GPU evaluation remain follow-up work for this version. Performance observations should identify the source revision, hardware, problem shape, backend and timing method.

## Author

Yu Gang (Yuvinci)
