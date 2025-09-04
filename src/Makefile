CC ?= $(shell command -v gcc 2>/dev/null || command -v clang 2>/dev/null)

# Detect compiler to set OpenMP flags
ifeq ($(findstring clang,$(notdir $(CC))),clang)
	OMPFLAGS = -fopenmp
else
	OMPFLAGS = -fopenmp
endif

# Portable by default (avoid Illegal instruction on older CPUs)
ARCHFLAGS ?= -march=x86-64 -mtune=generic
CFLAGS ?= -O3 $(ARCHFLAGS) $(OMPFLAGS) -Wall -Wextra
LDFLAGS ?= $(OMPFLAGS) -lm
PREFIX ?= $(CURDIR)/..
BIN_DIR := $(PREFIX)/bin
SRC := cpu_bench.c
BIN := $(BIN_DIR)/cpu_bench
HOSTNAME ?= $(shell hostname -s 2>/dev/null || hostname)
NATIVE_BIN := $(BIN_DIR)/bench-$(HOSTNAME)

all: bench

# Force portable or native builds
portable:
	$(MAKE) ARCHFLAGS='-march=x86-64 -mtune=generic' bench

native:
	$(MAKE) ARCHFLAGS='-march=native' bench

bench: $(BIN)

$(BIN): $(SRC) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

native-host: $(NATIVE_BIN)

$(NATIVE_BIN): $(SRC) | $(BIN_DIR)
	$(CC) -O3 -march=native $(OMPFLAGS) -Wall -Wextra -o $@ $< $(LDFLAGS)

$(BIN_DIR):
	mkdir -p $@

clean:
	rm -rf $(BIN_DIR)

.PHONY: all bench clean native native-host portable
