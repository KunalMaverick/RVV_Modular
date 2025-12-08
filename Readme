# RVV Modular Arithmetic & Matrix Multiplication

High-performance modular arithmetic and matrix multiplication using the RISC-V Vector Extension (RVV) with Barrett reduction for fast modular multiplication.

Designed for:
* Cryptographic workloads (NTT, HE, lattices)
* RVV benchmarking
* Accelerator validation
* Hardware/Software co-design

## Project Structure

```
test.c
uint_mod_arith_rvv.c
uint_mod_arith_rvv.h
uintarith_rvv.h
util.h
util_modular.h
Makefile
README.md
```

## Features

* Vectorized modular add, sub, and negation using RVV
* Vectorized modular multiplication using 128-bit multiply + Barrett reduction
* Dense RVV matrix multiplication
* RVV modular matrix multiplication
* Scalar golden reference for correctness checking
* Verified execution using the Spike RISC-V simulator

## Build Requirements

You must have the following installed:
* RISC-V GCC with RVV support
* Spike RISC-V simulator
* Proxy kernel (`pk`)

## How to Build

```bash
make
```

This produces:

```
test.elf
```

## How to Run on Spike

```bash
make run
```

Which executes:

```bash
spike --isa=rv64gcv pk test.elf
```

## What the Program Does

When executed, the program:
1. Tests RVV modular addition, subtraction, and negation
2. Tests RVV modular multiplication using Barrett reduction
3. Runs scalar versus RVV matrix multiplication
4. Runs full RVV modular matrix multiplication
5. Verifies correctness against scalar reference implementations
6. Prints timing and output samples

## Cleaning the Build

```bash
make clean
```

## Architecture Summary

* ISA: RV64GCV
* Data type: `uint64`
* Vector length: runtime determined via `vsetvl`
* Modular reduction: Barrett reduction (128-bit)

## Simulation Flow

```
test.elf → pk → Spike RVV simulator
```

## Use Cases

* RVV performance benchmarking
* Modular arithmetic kernel validation
* Cryptographic primitive testing
* Accelerator hardware/software co-design
* Computer architecture research

## License

Free to use for academic and research purposes.