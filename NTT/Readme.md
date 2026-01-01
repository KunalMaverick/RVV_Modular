# RVV NTT / INTT (Scalar + Vector Verified)

This repository contains a **fully working and verified implementation of the
Number Theoretic Transform (NTT) and its inverse (INTT)** using:

- **Scalar reference kernels** (ground truth)
- **RISC-V Vector (RVV) kernels** using RVV 0.7.1 intrinsics
- **Barrett reduction (`mulhu`)** for modular arithmetic

Both implementations are tested end-to-end such that:

and verified on **Spike RV64GCV**.

---

## 📂 File Overview

| File | Description |
|----|----|
| `ntt-rvv.c` | RVV forward NTT (Korn–Lambiotte) + RVV inverse NTT (Pease), with scalar fallback |
| `ntt-rvv.h` | Public API for NTT / INTT kernels |
| `ntt_test.c` | Test harness running **scalar + RVV** NTT/INTT and verifying correctness |

---

## 🔢 Parameters (Test Configuration)

| Parameter | Value |
|--------|------|
| Transform size | `N = 8` |
| Modulus | `q = 17` |
| Primitive root | `ω = 9` |
| Inverse root | `ω⁻¹ = 2` |
| Modular inverse of N | `N⁻¹ = 15 mod 17` |

> The small parameters are chosen for **debuggability and correctness validation**.
> The implementation generalizes to larger power-of-two sizes.

---

## 🧠 Algorithms Used

### Scalar Path (Reference)
- Cooley–Tukey forward NTT
- Gentleman–Sande inverse NTT
- Uses `% mod` arithmetic
- Serves as **ground truth**

### RVV Path
- **Forward NTT:** Korn–Lambiotte formulation
- **Inverse NTT:** Pease algorithm
- Uses:
  - RVV vector intrinsics
  - `mulhu`-based Barrett reduction
  - vector gathers, strided loads/stores
  - auxiliary buffers for stage swapping

> The RVV output layout differs from the scalar NTT (this is expected).
> Correctness is validated by invertibility, not output matching.

---

## ⚠️ Important Detail: Barrett Preconditioning

The RVV kernels rely on **Barrett reduction**, which requires **preconditioned constants**:


This must be done for:
- Roots of unity
- Inverse roots
- **N⁻¹ (critical for INTT correctness)**

Failing to precondition `N⁻¹` will cause RVV INTT to fail even if scalar passes.

---

## ▶️ How to Build & Run (Spike)

### Compile
```bash
riscv64-unknown-elf-gcc \
  -march=rv64gcv -mabi=lp64d \
  ntt_test.c ntt-rvv.c \
  -O2 -o ntt_test.elf
spike --isa=rv64gcv --varch=vlen:256,elen:64 pk ntt_test.elf


