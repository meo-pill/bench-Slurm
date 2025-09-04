CC ?= $(shell command -v gcc 2>/dev/null || command -v clang 2>/dev/null)

# Detect compiler to set OpenMP flags
ifeq ($(findstring clang,$(notdir $(CC))),clang)
	OMPFLAGS = -fopenmp
else
	OMPFLAGS = -fopenmp
endif

CFLAGS ?= -O3 -march=native $(OMPFLAGS) -Wall -Wextra
LDFLAGS ?= $(OMPFLAGS) -lm
PREFIX ?= $(CURDIR)
BIN_DIR := $(PREFIX)/bin
SRC := cpu_bench.c
BIN := $(BIN_DIR)/cpu_bench

all: bench

bench: $(BIN)

$(BIN): $(SRC) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BIN_DIR):
	mkdir -p $@

clean:
	rm -rf $(BIN_DIR)

.PHONY: all bench clean
