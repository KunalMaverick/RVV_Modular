# ============================================================
#  RVV Modular Arithmetic + Matrix Multiply Build
# ============================================================

# Toolchain
CC = riscv64-unknown-elf-gcc
SPIKE = spike
PK = pk

# Architecture Flags
ARCH = -march=rv64gcv -mabi=lp64d

# Optimization
CFLAGS = -O2 $(ARCH)

# Files
SRC = test.c uint_mod_arith_rvv.c
OUT = test.elf

# ============================================================
#  Build Targets
# ============================================================

all: $(OUT)

$(OUT): $(SRC)
	@$(CC) $(CFLAGS) $(SRC) -o $(OUT)

run: $(OUT)
	@$(SPIKE) --isa=rv64gcv $(PK) $(OUT)

clean:
	@rm -f test.elf

# ============================================================
#  Convenience Targets
# ============================================================

rebuild: clean all

.PHONY: all run clean rebuild
