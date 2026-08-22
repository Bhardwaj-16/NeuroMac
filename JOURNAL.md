----------

## NeuroMAC
author: Bhardwaj Prasad Sutara
description: A custom FPGA-based INT8 neural-network accelerator implemented as an AXI-connected 
Verilog IP core on a Zynq-7000, integrated with an ARM processor to accelerate MAC operations for an 
MNIST MLP.
created_at: "2026-08-21"

# August 21: The board & the project Scope

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

**Total time spent: 1.5 hours**


# August 22: Hardware Design & IP Generation

**sigh** This is probably going to be the most time-consuming part. But then again, this is the actual hardware.

<img width="1440" height="574" alt="hardware" src="https://github.com/user-attachments/assets/dc211724-974a-4e83-b7ab-3b1dfc143c78" />

Today was the point where the project moved from an idea into actual hardware design.

The objective was to take the core computation required by the neural-network workload and implement it as a custom hardware accelerator in Verilog, then expose that accelerator to the ARM processor inside the Zynq SoC through an AXI4-Lite interface.(Academia-ish words. I Know)

The final goal is to create something that behaves like a real processor peripheral:
**ARM Processor → AXI4-Lite → Custom NPU IP → Hardware Compute → AXI4-Lite → ARM Processor**

That meant I had to deal with several different layers simultaneously:

1. Designing the arithmetic hardware.
2. Designing the control FSM.
3. Handling signed arithmetic and data widths.
4. Creating memory-mapped control/data registers.
5. Implementing the AXI4-Lite slave interface.
6. Creating status signals such as `busy` and `done`.
7. Adding a hardware ReLU activation stage.
8. Verifying the behavior in simulation.
9. Packaging/integrating the IP into the Vivado block design.

---

## Why build the accelerator in hardware?

The neural network ultimately needs to perform a large number of arithmetic operations.

At the most basic level, neural-network inference repeatedly performs operations such as:

```text
multiplication
+
addition
+
activation
```

On a normal processor, these operations execute sequentially through the CPU's instruction pipeline.
The purpose of an accelerator is to move part of this workload into dedicated digital logic(NPU).


**The processor hands over the operands to the NPU instead of doing the process sequentially by itself. The NPU is capable of doing operations in parallel**

The FPGA fabric then performs the computation using dedicated hardware and eventually exposes the result back to the processor.
That is the fundamental idea behind the NPU in this project.
---

# Starting with the arithmetic core

The first part of the design was the actual computation.

I defined the matrix operands as signed 8-bit values:

```verilog
localparam DATA_WIDTH = 8;
```

The inputs are therefore represented by:

```text
a00
a01
a10
a11

b00
b01
b10
b11
```

with each operand being:

```verilog
reg signed [DATA_WIDTH-1:0]
```

This was important because neural-network data can contain negative values, particularly when weights are represented using signed fixed-point/integer formats.
Using unsigned arithmetic would have caused incorrect results whenever negative operands were involved.

---

# Result width

I initially had to think carefully about how many bits were required for the result.

An 8-bit signed value can represent approximately:

```text
-128 → +127
```

Multiplying two signed 8-bit numbers can therefore require more bits than the original operands.

I allocated:

```verilog
localparam RESULT_WIDTH = 17;
```

and stored the results in signed registers:

```verilog
reg signed [RESULT_WIDTH-1:0] c00, c01, c10, c11;
```

This gave the arithmetic pipeline additional headroom.
This is extremely important once real hardware is involved.(I wanna try things in simulations as much as possible because I dont want to encounter any errors after buying the physical board)
---

# Designing the control FSM

Once the arithmetic portion existed, I needed a way to control it.
I didn't want the accelerator to simply perform everything in one enormous combinational block.

Instead, I divided the operation into explicit states:

```text
IDLE
  ↓
LOAD
  ↓
COMPUTE
  ↓
ACTIVATE
  ↓
STORE
  ↓
DONE
  ↓
IDLE
```

These became:

```verilog
S_IDLE
S_LOAD
S_COMPUTE
S_ACTIVATE
S_STORE
S_DONE
```

The FSM gives the accelerator a predictable sequence of operations.

---

# IDLE state

The accelerator begins in:

```verilog
S_IDLE
```

Here:

```verilog
npu_busy <= 1'b0;
```

The accelerator is waiting for the processor to request a computation.
I also implemented a rising edge detector for the start signal.

Instead of continuously triggering while the start bit remained high, the design tracks the previous value:

```verilog
reg prev_start;
```

and generates:

```verilog
assign start_rising = slv_reg0[0] & ~prev_start;
```

This means a computation begins when the start control bit transitions from:

```text
0 → 1
```

rather than simply remaining at `1`.

This prevents the accelerator from repeatedly restarting just because software left the start bit asserted.

---

# LOAD state

Once a start request is detected, the FSM moves to:

```verilog
S_LOAD
```

The purpose of this state is to copy the operands from the AXI visible registers into internal hardware registers.

For example:

```verilog
a00 <= slv_reg2[DATA_WIDTH-1:0];
a01 <= slv_reg2[DATA_WIDTH*2-1:DATA_WIDTH];
```

The same process occurs for the remaining elements of A and B.
This separation is useful because the AXI registers are essentially the interface between software and hardware.
The internal registers become the accelerator's working operands.

The architecture is roughly:

```text
ARM
 |
 | write registers
 v
AXI registers
 |
 | LOAD
 v
Internal NPU registers
 |
 v
Compute hardware
```

---

# COMPUTE state

The next state is:

```verilog
S_COMPUTE
```

The current implementation performs four signed multiplication operations:

```verilog
c00 <= $signed(a00) * $signed(b00);
c01 <= $signed(a01) * $signed(b01);
c10 <= $signed(a10) * $signed(b10);
c11 <= $signed(a11) * $signed(b11);
```

The important thing here is that these are hardware multipliers.
The FPGA synthesis tool can map these operations onto the FPGA's available arithmetic resources, potentially using DSP slices depending on the target device and synthesis configuration.

### Important scope note

At this stage, the implementation is better described as a **parallel multiply engine / MAC-oriented compute primitive**, rather than a complete 2×2 matrix multiplication unit.

A mathematically complete matrix multiplication would require accumulation, for example:

```text
C00 = A00×B00 + A01×B10
C01 = A00×B01 + A01×B11
C10 = A10×B00 + A11×B10
C11 = A10×B01 + A11×B11
```

The current hardware performs the multiplication portion in parallel. This is intentional as an incremental hardware design step and leaves room for expanding the compute engine later.
---

# Hardware ReLU activation

After computation, the FSM enters:

```verilog
S_ACTIVATE
```

This is where I implemented ReLU directly in hardware.

The ReLU operation is conceptually:

```text
ReLU(x) = max(0, x)
```

Instead of sending the result back to software and asking the ARM processor to perform the activation, the FPGA performs the operation.

The implementation checks the sign bit:

```verilog
c00[RESULT_WIDTH-1]
```

and, if ReLU is enabled and the result is negative, replaces it with zero.

For example:

```verilog
c00_act <= (slv_reg0[8] && c00[RESULT_WIDTH-1]) ? 0 : c00;
```

The same logic is applied to all four outputs.
This is important conceptually because it demonstrates that the accelerator is not merely an arithmetic peripheral.

It is beginning to implement an actual neural-network computation pipeline:

```text
Multiply → Activation
```

rather than relying entirely on the processor.

---

# STORE state

The FSM then moves into:

```verilog
S_STORE
```

Here the activation results are copied into the output registers:

```verilog
slv_reg6
slv_reg7
slv_reg8
slv_reg9
```

These registers are visible through the AXI interface.
Therefore, the ARM processor can eventually read the results through memorymapped addresses.

The overall data path becomes:

```text
Software
   ↓
AXI write
   ↓
Input registers
   ↓
NPU internal registers
   ↓
Multiplier
   ↓
ReLU
   ↓
Output registers
   ↓
AXI read
   ↓
Software
```

---

# DONE and BUSY status

The accelerator also needs a way to tell software what it is doing.

I therefore created:

```verilog
npu_busy
npu_done
```

and exposed them through the status register:

```verilog
slv_reg1
```

with:

```text
bit 0 = busy
bit 1 = done
```

This gives software a simple interface.

Conceptually:

```text
write inputs
↓
write START
↓
wait while BUSY = 1
↓
wait for DONE
↓
read outputs
```

---

# Designing the AXI4-Lite interface

This was probably one of the more complicated sections.
The accelerator cannot simply expose Verilog registers directly to the ARM processor.
The Zynq processing system communicates with peripherals through standardized interfaces.

I wrapped the compute engine inside an:

**AXI4-Lite slave interface.**

The AXI interface provides:

* write address channel
* write data channel
* write response channel
* read address channel
* read data channel

The processor can therefore access the accelerator as if it were a normal memory mapped peripheral.

---

# Register map

I created a memory-mapped register structure.
(AI was used to create the tabular structure)

The conceptual register map is:

| Register    | Purpose                |
| ----------- | ---------------------- |
| `slv_reg0`  | Control                |
| `slv_reg1`  | Status                 |
| `slv_reg2`  | A00/A01                |
| `slv_reg3`  | A10/A11                |
| `slv_reg4`  | B00/B01                |
| `slv_reg5`  | B10/B11                |
| `slv_reg6`  | C00                    |
| `slv_reg7`  | C01                    |
| `slv_reg8`  | C10                    |
| `slv_reg9`  | C11                    |
| `debug_reg` | Debug/state visibility |

The control register contains the start and ReLU enable controls.
The status register reports whether the accelerator is busy or finished.
---

# Debug register

I also added a debug register:

```verilog
debug_reg
```

with recognizable hexadecimal markers:

```text
AAAAAAAA
BBBBBBBB
ACACACAC
DDDDDDDD
EEEEEEEE
```

These correspond to different stages of the FSM.

For example:

```text
AAAAAAAA → LOAD
BBBBBBBB → COMPUTE
ACACACAC → ACTIVATE
DDDDDDDD → STORE
EEEEEEEE → DONE
```

When debugging hardware, it can be extremely useful to expose internal state information through a register rather than relying entirely on waveform inspection.
Also it would be easier for me to know the STATE of the accelerator
---

# Reset verification

I verified that reset returned the accelerator to a known state.

The reset sequence clears:

```text
FSM state
busy
done
input registers
output registers
internal operands
debug register
```

The accelerator should therefore start from:

```text
S_IDLE
BUSY = 0
DONE = 0
```

rather than accidentally retaining values from an earlier operation.
This is especially important for FPGA hardware because the simulation may begin in a state that doesn't correspond to the intended post-reset behavior unless reset is explicitly applied.

---

# Integration into Vivado

Once the RTL behavior was sufficiently verified, the next step was to integrate the accelerator into the actual Zynq system.

The architecture became approximately:
(Again, AI Usage for the diagram)

```text
                    Zynq-7000
              ┌───────────────────┐
              │                   │
              │   ARM Cortex-A9   │
              │                   │
              └─────────┬─────────┘
                        │
                    AXI Interconnect
                        │
                        ▼
              ┌───────────────────┐
              │   Custom NPU IP   │
              │                   │
              │ AXI4-Lite Slave   │
              │       │           │
              │       ▼           │
              │ Control Registers │
              │       │           │
              │       ▼           │
              │      FSM          │
              │       │           │
              │       ▼           │
              │ Multiply Engine   │
              │       │           │
              │       ▼           │
              │      ReLU         │
              │       │           │
              │       ▼           │
              │ Output Registers  │
              └───────────────────┘
```

The custom IP was then placed into the Vivado block design alongside the Zynq processing system.
This was the point where the design stopped being an isolated Verilog module and became an actual SoC peripheral.

---

By the end of the work, I had created the basic hardware/software interface for the accelerator.

The processor can conceptually:

```text
1. Write input data
2. Configure ReLU
3. Assert START
4. Accelerator loads operands
5. Hardware performs parallel multiplication
6. Hardware performs ReLU
7. Results are stored
8. DONE becomes asserted
9. Processor reads the results
```
---


Also, the current compute stage performs four parallel multiplications, but it does **not yet perform the complete 2×2 matrix multiply-accumulate operation**.

The next hardware iteration should therefore introduce the accumulation stage:

```text
C00 = A00×B00 + A01×B10
C01 = A00×B01 + A01×B11
C10 = A10×B00 + A11×B10
C11 = A10×B01 + A11×B11
```

That would turn the current multiply-oriented engine into a genuine small matrix-multiplication/MAC engine.

Oh man and the time(I didnt even document the errors and debugging)
**Total time spent: 10 hours**
