# Topic: CNN acceleration with OpenMP

## Introduction
### Whats CNN
#### Area of implementation
#### Methods to accelerate
Basically to reduce computation complexity/cut off graphs, to increase computation speed or reduce data flow.  
Some technologies concentrate on optimizing computational graphs and minimizing the data transferred, employing enhanced encoding and decoding techniques, such as TFRecord. Others aim to reduce computational overhead, utilizing tools like TensorRT and the XLA Compiler for graph optimization. However, our project prioritizes reducing processing time for a specified workload through enhanced parallelism and effective data transfer management.

1. **Parallelism**:
   + **Data Parallelism**: TensorFlow can distribute the inference workload of a CNN across multiple GPUs or other processors. It splits the input data into smaller batches and processes them simultaneously on different devices. This type of parallelism increases throughput and reduces overall inference time.
   + **Model Parallelism**: In some cases, especially when dealing with very large models that cannot fit into the memory of a single GPU, TensorFlow allows splitting the model itself across multiple devices. Different layers or parts of the model can be processed on different devices, utilizing the computational power of each device optimally.
   + **Pipeline Execution**: For models with sequential dependencies (like CNNs), TensorFlow can use pipeline execution, where different stages of the model are processed on different GPUs in a pipelined manner. This ensures that all GPUs are utilized efficiently, reducing idle times and maximizing throughput.
2. **Efficient Memory Management**:
    - **Memory Pooling**: TensorFlow optimizes memory allocation by pooling memory. Instead of repeatedly allocating and deallocating memory for tensors during the lifecycle of model inference, TensorFlow uses a pool of pre-allocated memory blocks. This reduces the overhead associated with memory management and speeds up tensor operations.
    - **Tensor Fusion**: TensorFlow can combine multiple small tensor operations into fewer, larger operations. This reduces the overhead of launching multiple kernel operations and leads to more efficient utilization of the computational resources. By doing so, it minimizes memory read/write operations which are often a bottleneck.
    - **Asynchronous Execution**: TensorFlow can perform certain operations asynchronously, especially I/O operations like loading data or writing outputs. This means that computation can overlap with data transfer or other preprocessing steps, effectively utilizing CPU and GPU resources without waiting for one task to complete before starting another.
3. **Optimized Kernels**:
    - **Custom CUDA Kernels**: For operations that are particularly intensive or common in CNNs, such as convolution and pooling, TensorFlow can use optimized CUDA kernels. These are specifically designed to maximize performance on NVIDIA GPUs, taking advantage of specific hardware features like shared memory and tensor cores.
    - **Kernel Autotuning**: TensorFlow can use XLA (Accelerated Linear Algebra) to automatically tune CUDA kernels based on the specific model and hardware configuration. This autotuning process finds the optimal configurations for thread blocks and grid sizes, which can significantly enhance performance.
4. **Advanced Scheduling**:
    - **Smart Operation Scheduling**: TensorFlow's runtime can intelligently schedule operations based on their computational requirements and the availability of hardware resources. This helps in minimizing idle times and ensures that operations that can be executed in parallel are scheduled simultaneously.

### Whats OpenMP

## Framework of the application
+ Based on MOJO-CNN(https://github.com/gnawice/mojo-cnn)
+ Dataset Mnist written digits(https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
  + Training dataset size 60k, 28\*28\*1
### CNN structure
How is the application running...
#### 5 layer

## Analysis 

### Main Analysis tools: Intel Vtune

Performance Analysis for Applications & Systems

Intel® VTune™ Profiler optimizes application performance, system performance, and system configuration for HPC, cloud, IoT, media, storage, and more.  
CPU, GPU, and FPGA: Tune the entire application’s performance―not just the accelerated portion.  
Multilingual: Profile SYCL*, C, C++, C#, Fortran, OpenCL™ code, Python*, Google Go* programming language, Java*, .NET, Assembly, or any combination of languages.  
System or Application: Get coarse-grained system data for an extended period or detailed results mapped to source code.  
Power: Optimize performance while avoiding power- and thermal-related throttling.

Notice That the system application and power monitor require additional driver, or there will be a limit on the stack size.

> if run in sudo mode, do "xhost +" before you start

### Test Env
#### CPU Info
+ Model Name: 12th Gen Intel(R) Core(TM) i9-12900H
+ cache size	: 24576 KB (24MB)
+ cpu core      : 14
+ Processor num : 20 
  + Intel Hyper-Threading Technology is only available on Performance-cores
+ Max Turbo Frequency : 5.00 GHz
+ Efficient-core Max Turbo Frequency : 3.80 GHz
+ clflush size	: 64 bytes
+ cache_alignment	: 64 bytes
+ address sizes	: 46 bits physical, 48 bits virtual
#### Mem Info
2\*  
Type: DDR5  
Speed: 4800 MT/s  
Configured Memory Speed: 4800 MT/s  
Volatile Size: 16 GB  

### Tables
+ Performance Snapshot
+ System Overview
+ Input Output Analysis
+ Threading
+ Memory Access
+ Memory Consumption

### metrics 
+ running time
+ | # of Threads | Running Time |      |
  | ------------ | ------------ | ---- |
  | 1            | 93.139       |      |
  | 5            | 53.787       |      |
  | 10           | 78.993       |      |
  | 20           | 90.149       |      |

  


## Hypnosis
+ (one or two bullet point)

