----------

## NeuroMAC
author: Bhardwaj Prasad Sutara
description: A custom FPGA-based INT8 neural-network accelerator implemented as an AXI-connected 
Verilog IP core on a Zynq-7000, integrated with an ARM processor to accelerate MAC operations for an 
MNIST MLP.
created_at: "2026-08-21"

# August 19: The board & the project Scope

Im thinking of using the Xilinx Zynq-7000 Z7-10 SoC based board. I found Zybo Z7-10 very intresting.
I wanna first explain the pros of this SoC. The Zynq architecture is convenient for demonstrating a 
custom hardware accelerator controlled by a CPU. Which exactly what this project aims for.
We'll we could also theoretically use just an FPGA. BUT, we'd need some external processor or soft 
CPU to control the accelerator.

The Zynq works somethign like this:
ARM CPU ──AXI──► FPGA logic
The chip already consists of an ARM Processor.

So the flow is:
ARM runs the C Program, C writes data into AXI registers, the registers control our Verilog NPU, then 
the NPU performs the multiplication in FPGA hardware, ARM reads the result and continues the neural 
network inference.

The Z7-10 is the smaller member of the Zybo Z7 family which has the SoC im planning to use. And this 
project is relatively fundamental and easy on compute. Just 8bit multiplications.

The question I'm trying to answer is,
`Can I move a computationally expensive part of neural-network inference from software running on a CPU into custom hardware?`

Because without a custom hardware, the flow basically is,
ARM CPU

784 inputs
    ↓
40 neurons
    ↓
MAC operations
    ↓
10 outputs

But with an NPU,
```
              ARM CPU
                 │
                 │ AXI
                 ▼
          ┌─────────────┐
          │  Custom NPU │
          │             │
          │ 4 × int8    │
          │ multipliers │
          └─────────────┘
                 │
                 ▼
             products
                 │
                 ▼
              ARM CPU

```
Some of the computation is physically implemented in FPGA logic.

* TLDR: This project implements a Neural Processing Unit (NPU) as a custom IP on a Xilinx Zynq-7000 series FPGA for accelerating MNIST digit recognition in Python, quantizes the weights, and accelerates the matrix multiplication operations on a custom hardware NPU. The NPU performs 2x2 Multiply-Accumulate (MAC) operations, a fundamental building block for neural network inference. Also as of now we aren't using a camera as I feel the board itself costs much that I dont feel like spending more money on integrating a camera.

** Total time spent: 1.5 hours
