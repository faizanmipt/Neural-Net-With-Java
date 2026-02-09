import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.CountDownLatch;

/**
 * Multi-threaded neural network forward pass benchmark.
 * 
 * This benchmark evaluates parallel dense layer computation performance
 * across varying thread counts, demonstrating scaling behavior from 
 * physical to logical CPU cores.
 */
public final class NetworkForwardPassPerformance {

    /** Represents a fully-connected neural network layer with trainable weights. */
    static final class DenseLayer {
        /** Number of input neurons */
        final int in, out;
        /** Weight matrix in column-major layout for cache-efficient access */
        final float[] weights; // column-major
        /** Bias terms added to each output neuron */
        final float[] bias;

        /**
         * Creates a dense layer with specified dimensions.
         * Weights are initialized using a deterministic pattern for reproducibility.
         * 
         * @param in  Number of input connections
         * @param out Number of output neurons
         */
        DenseLayer(int in, int out) {
            this.in = in;
            this.out = out;
            this.weights = new float[in * out];
            this.bias = new float[out];

            // Initialize weights with a structured pattern
            // Values range approximately from -0.06 to 0.06
            for (int o = 0; o < out; o++) {
                for (int i = 0; i < in; i++) {
                    weights[i * out + o] = ((i % 13) - 6) * 0.01f;
                }
            }
        }
    }

    /**
     * Neural network composed of multiple dense layers.
     * Uses a single contiguous activations array with offset indexing for memory efficiency.
     */
    static final class Network {
        /** Number of hidden layers (excluding input layer) */
        final int layers;
        /** Array of dense layers forming the network */
        final DenseLayer[] layersArr;
        /** Flattened activation storage for all layers and batch samples */
        final float[] activations; // single flattened array
        /** Starting index offset for each layer in the activations array */
        final int[] layerOffset;   // start index of each layer in activations
        /** Neuron count per layer */
        final int[] layerSizes;    // size of each layer
        /** Number of samples processed per batch */
        final int batch;

        /**
         * Constructs a neural network with given layer sizes.
         * Pre-allocates a single contiguous activation array.
         * 
         * @param sizes Array of neuron counts per layer
         * @param batch Number of samples in each batch
         */
        Network(int[] sizes, int batch) {
            this.layers = sizes.length - 1;
            this.batch = batch;
            this.layersArr = new DenseLayer[layers];
            this.layerOffset = new int[layers + 1];
            this.layerSizes = sizes.clone();

            // Initialize dense layers
            for (int l = 0; l < layers; l++)
                layersArr[l] = new DenseLayer(sizes[l], sizes[l + 1]);

            // Apply small positive bias to output layer
            for (int i = 0; i < layersArr[layers - 1].bias.length; i++)
                layersArr[layers - 1].bias[i] = 0.1f;

            // Compute total activation storage requirements and layer offsets
            // Each layer needs: batch * layerSize float values
            int totalSize = 0;
            for (int l = 0; l <= layers; l++) {
                layerOffset[l] = totalSize;
                totalSize += batch * sizes[l];
            }
            activations = new float[totalSize];
        }

        /**
         * Computes the flat array index for a specific layer and batch sample.
         * 
         * @param layer      Layer index (0 = input layer)
         * @param batchIndex Sample index within the batch
         * @return Flattened array index for the activation value
         */
        int getLayerIndex(int layer, int batchIndex) {
            return layerOffset[layer] + batchIndex * layerSizes[layer];
        }
    }

    /**
     * Executes a parallel forward pass through the entire network.
     * 
     * Thread work distribution:
     * - Each thread processes a contiguous chunk of batch samples
     * - Layers are processed sequentially within each thread
     * - Block-wise computation optimizes cache utilization
     * 
     * @param net    Network to forward propagate
     * @param batch  Batch size (for reference, actual size from net.batch)
     * @param pool   Thread executor for parallel work
     * @param threads Number of concurrent worker threads
     * @param block  Size of computation blocks for cache efficiency
     * @throws InterruptedException if threads are interrupted during execution
     */
    static void forwardBatch(Network net, int batch, ExecutorService pool, int threads, int block)
            throws InterruptedException {
        // Synchronization barrier ensuring all threads complete before returning
        CountDownLatch latch = new CountDownLatch(threads);

        // Launch worker threads, each handling a subset of the batch
        for (int t = 0; t < threads; t++) {
            final int thread = t;
            pool.execute(() -> {
                // Determine this thread's batch chunk (last chunk may be smaller)
                int chunk = (batch + threads - 1) / threads;
                int start = thread * chunk;
                int end = Math.min(batch, start + chunk);

                // Per-thread accumulation buffer, reused for all samples
                float[] sum = new float[block];

                // Process each layer in sequence
                for (int l = 0; l < net.layers; l++) {
                    DenseLayer layer = net.layersArr[l];
                    int inSize = layer.in;
                    int outSize = layer.out;
                    float[] w = layer.weights;
                    float[] b = layer.bias;

                    // Process assigned batch samples
                    for (int bIdx = start; bIdx < end; bIdx++) {
                        // Calculate activation offsets using helper method
                        int inOffset = net.getLayerIndex(l, bIdx);
                        int outOffset = net.getLayerIndex(l + 1, bIdx);

                        // Block-wise computation for output neurons
                        int o = 0;
                        // Process full blocks
                        for (; o + block <= outSize; o += block) {
                            // Initialize with bias values
                            for (int i = 0; i < block; i++)
                                sum[i] = b[o + i];

                            // Accumulate weighted inputs
                            for (int i = 0; i < inSize; i++) {
                                float x = net.activations[inOffset + i];
                                int wOffset = i * outSize + o;
                                for (int j = 0; j < block; j++)
                                    sum[j] += w[wOffset + j] * x;
                            }

                            // Apply ReLU activation
                            for (int i = 0; i < block; i++)
                                net.activations[outOffset + o + i] = Math.max(sum[i], 0f);
                        }

                        // Handle remaining neurons (when outSize % block != 0)
                        for (; o < outSize; o++) {
                            float s = b[o];
                            for (int i = 0; i < inSize; i++)
                                s += w[i * outSize + o] * net.activations[inOffset + i];
                            net.activations[outOffset + o] = Math.max(s, 0f);
                        }
                    }
                }

                // Signal completion of this thread's work
                latch.countDown();
            });
        }

        // Wait for all threads to finish
        latch.await();
    }

    /**
     * Formats and outputs benchmark timing results.
     * 
     * @param label   Descriptive label for the configuration
     * @param start   Start time in nanoseconds
     * @param end     End time in nanoseconds
     * @param samples Total number of samples processed
     */
    static void report(String label, long start, long end, long samples) {
        double totalUs = (end - start) / 1_000.0;
        double usPerSample = totalUs / samples;
        System.out.printf("%-14s : %.3f µs/sample%n", label, usPerSample);
    }

    /**
     * Main entry point - runs multi-core scaling benchmarks.
     * 
     * Execution phases:
     * 1. Warmup (5 iterations): Triggers JIT compilation
     * 2. Physical cores: Tests thread counts from 1 to physical core count
     * 3. Logical cores: Tests oversubscription with SMT threads
     * 
     * @param args Command line arguments (unused)
     * @throws Exception if execution fails
     */
    public static void main(String[] args) throws Exception {
        // Network topology: 1024 -> 1024 -> 512 -> 128 -> 10
        int[] sizes = {1024, 1024, 512, 128, 10};
        int batch = 256;
        int warmup = 5;
        int iters = 10;
        int block = 32;

        // Create network instance
        Network net = new Network(sizes, batch);

        // Initialize input layer with structured pattern
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < sizes[0]; i++)
                net.activations[net.getLayerIndex(0, b) + i] = (i % 7) * 0.1f;

        // Configure thread pool based on physical cores
        int logical = Runtime.getRuntime().availableProcessors();
        int physicalCap = Math.max(1, logical / 2);
        ExecutorService pool = Executors.newFixedThreadPool(physicalCap);

        System.out.println("Logical cores : " + logical);
        System.out.println("Physical cap  : " + physicalCap + "\n");

        // Warmup phase - JIT compilation and cache population
        for (int i = 0; i < warmup; i++)
            forwardBatch(net, batch, pool, physicalCap, block);

        // Scale up through physical cores
        System.out.println("=== Scaling up to physical cores ===");
        for (int t = 1; t <= physicalCap; t *= 2) {
            long start = System.nanoTime();
            for (int i = 0; i < iters; i++)
                forwardBatch(net, batch, pool, t, block);
            long end = System.nanoTime();
            report("Threads=" + t, start, end, iters * batch);
        }

        // Test oversubscription with logical cores
        System.out.println("\n=== SMT / logical cores / oversubscription ===");
        for (int t = physicalCap + 1; t <= logical; t++) {
            long start = System.nanoTime();
            for (int i = 0; i < iters; i++)
                forwardBatch(net, batch, pool, t, block);
            long end = System.nanoTime();
            report("Threads=" + t, start, end, iters * batch);
        }

        // Compute validation checksum
        double guard = 0;
        int outLayer = net.layers;
        for (int b = 0; b < batch; b++) {
            int outOffset = net.getLayerIndex(outLayer, b);
            for (int i = 0; i < sizes[outLayer]; i++)
                guard += net.activations[outOffset + i];
        }
        System.out.println("\nGuard: " + guard);

        pool.shutdown();
    }
}
