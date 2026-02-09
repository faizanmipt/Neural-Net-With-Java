# Neural Network Forward Propagation Performance Test

A high-performance Java implementation for measuring multi-threaded neural network forward pass throughput across varying thread counts. This test demonstrates scaling behavior from physical to logical CPU cores using a fully-connected network architecture.

## Overview

This project evaluates parallel dense layer computation performance by processing batches of input samples through a multi-layer neural network. The implementation focuses on:

- **Thread-level parallelism**: Batch samples distributed across worker threads
- **Cache-efficient computation**: Block-wise processing strategy (32-element blocks)
- **JVM optimization**: Leverages JIT compilation and auto-vectorization
- **Performance profiling**: Measures throughput across different thread configurations

## Network Architecture

The test uses a 4-layer fully-connected network:

```
Input (128) → Dense (256, ReLU) → Dense (512, ReLU) → Dense (1024, ReLU) → Dense (1024, ReLU) → Output (10)
```

### Layer Configuration

| Layer | Input Size | Output Size | Parameters |
|-------|------------|-------------|------------|
| 0 | 128 | 256 | 32,768 |
| 1 | 256 | 512 | 131,072 |
| 2 | 512 | 1,024 | 524,288 |
| 3 | 1,024 | 1,024 | 1,048,576 |

**Total parameters**: ~1.7 million weights

## Requirements

- **Java**: JDK 8 or higher
- **Memory**: Minimum 64MB heap space
- **CPU**: Multi-core processor recommended

## Building

```bash
# Compile the Java source file
javac NetworkForwardPassPerformance.java

# Run the test
java NetworkForwardPassPerformance
```

## Usage

The test executes automatically with no command-line arguments required.

### Configuration

Modify the following constants in the `main()` method to customize the test:

```java
int[] sizes = {1024, 1024, 512, 128, 10};  // Network topology
int batch = 256;                             // Samples per batch
int warmup = 5;                              // JIT warmup iterations
int iters = 10;                              // Measured iterations
int block = 32;                              // Block size (cache optimization)
```

## Output Explanation

### Console Output

```
Logical cores : 20
Physical cap  : 10

=== Scaling up to physical cores ===
Threads=1      : 563.337 µs/sample
Threads=2      : 285.138 µs/sample
Threads=4      : 169.718 µs/sample
Threads=8      : 119.869 µs/sample

=== SMT / logical cores / oversubscription ===
Threads=11     : 101.211 µs/sample
Threads=12     : 103.019 µs/sample
Threads=13     : 100.929 µs/sample
Threads=14     : 104.416 µs/sample
Threads=15     : 102.321 µs/sample
Threads=16     : 107.247 µs/sample
Threads=17     : 99.838 µs/sample
Threads=18     : 106.003 µs/sample
Threads=19     : 101.569 µs/sample
Threads=20     : 93.294 µs/sample

Guard: 256.00000381469727
```

### Metrics Explained

| Metric | Description |
|--------|-------------|
| `Logical cores` | Total available processors (including hyper-threads) |
| `Physical cap` | Thread limit (logical / 2) for compute-bound work |
| `Threads=N` | Test configuration with N threads |
| `µs/sample` | Microseconds per sample (lower is better) |
| `Guard` | Validation checksum for correctness verification |

## Architecture

### Key Components

1. **DenseLayer**: Fully-connected layer with column-major weight storage
2. **Network**: Manages multiple layers with contiguous activation storage
3. **forwardBatch()**: Parallel forward pass implementation

### Memory Layout

- **Weights**: Column-major format (`weights[i * out + o]`)
- **Activations**: Flat contiguous array with offset indexing
- **Block size**: 32 elements (128 bytes) aligns with cache lines and SIMD widths

### Threading Model

```
Batch (256 samples) divided among threads:
├── Thread 1: Samples 0-63
├── Thread 2: Samples 64-127
├── Thread 3: Samples 128-191
└── Thread 4: Samples 192-255
```

Each thread processes all layers for its assigned samples before synchronization.

## Performance Characteristics

### Expected Behavior

1. **Linear scaling** up to physical core count
2. **Diminishing returns** at logical core counts (hyper-threading)
3. **JIT warmup effect**: First iterations slower due to compilation

### Factors Affecting Performance

- **CPU architecture**: Core count, cache size, SIMD width
- **JVM implementation**: HotSpot, OpenJ9 differ in optimization strategies
- **Operating System**: Thread scheduling priorities
- **Memory bandwidth**: Contention at high thread counts

## Technical Details

### ReLU Activation

The Rectified Linear Unit activation function:

```
ReLU(x) = max(0, x)
```

Implemented using `Math.max()` for JIT intrinsification support.

### Numerical Precision

- **Data type**: 32-bit float (IEEE 754 single precision)
- **Memory efficiency**: 4 bytes per element
- **SIMD throughput**: AVX2 processes 8 floats per instruction

### Cache Optimization

Block size of 32 elements (128 bytes) optimizes:
- L1 cache utilization (32KB per core)
- Cache line utilization (64 bytes)
- SIMD vectorization alignment

## Validation

The final checksum (`Guard`) verifies computational correctness:
- Consistent across thread configurations
- Detects race conditions
- Confirms deterministic execution

## Performance Tuning

For different hardware configurations:

1. **Adjust block size** based on cache size:
   ```java
   int block = 32;  // Try 16, 32, or 64
   ```

2. **Modify thread pool** for NUMA systems:
   ```java
   int physicalCap = Math.max(1, logical / 2);  // Current
   // Consider: Use Runtime.getRuntime().availableProcessors() directly
   ```

3. **Increase batch size** for memory-bound systems:
   ```java
   int batch = 256;  // Larger batches amortize overhead
   ```

## Troubleshooting

### Low Thread Scaling

- Ensure JVM warmup completes before measurement
- Check CPU frequency scaling (disable turbo boost)
- Verify process priority (use nice or renice)

### High Variance

- Increase iteration count (`iters`)
- Reduce system load during test
- Use CPU pinning (taskset on Linux)

### Memory Issues

- Reduce batch size if OutOfMemoryError occurs
- Increase heap size: `java -Xmx256m NetworkForwardPassPerformance`

## Contributing

Improvements welcome:

1. Additional network topologies
2. Different activation functions (Sigmoid, Tanh, Leaky ReLU)
3. Layer normalization support
4. Test comparison with native BLAS libraries


