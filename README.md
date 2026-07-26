# Bare-Metal Assembly Neural Network: Solving XOR from Scratch

This project was born out of an interest to learn assembly language and build a neural network from scratch. Initially, I wanted to write it in C, as plenty of Python implementations already exist online. But then I saw someone write one in assembly, and I thought to myself: *how bare-metal can it get from there!*

To summarize, this is my first time writing a neural network from scratch, and also my first time programming in a non-PIC MCU assembly language (x86-64).

The code itself uses a Multi-Layer Perceptron architecture consisting of exactly 3 neurons to successfully perform the non-linear XOR operation.

## The Architecture

I created several subroutines to mimic an OOP-style structure. Essentially, they are broken down into:

* **`forward_pass`**: Performs the math and ReLU activation for each neuron, satisfying the equation $y = w_n \cdot x_n + w_{n+1} \cdot x_{n+1} + b$.
* **`errors`**: Calculates the overall mistake using the operation `Target - Result`.
* **`backpropagate`**: Performs the learning part of the model. Each weight and bias is adjusted relative to the computed hidden error and the ReLU derivative.
* **`print`**: A custom subroutine to convert the floating-point fractions into ASCII to output the results to the terminal.

---

### 1. The Forward Pass

In this section, a single neuron's computation looks like the following snippet.

For example, in Neuron 1:

```nasm
    ; Neuron 1 computations
    movss xmm0, dword [rel w11]        ; Load w11 into xmm0
    mulss xmm0, dword [rsi]            ; Multiply w11 with input 1

    movss xmm1, dword [rel w12]        ; Load w12 into xmm1
    mulss xmm1, dword [rsi + 4]        ; Multiply w12 with input 2
    
    addss xmm0, xmm1                   ; Add the two products together
    addss xmm0, dword [rel b1]         ; Add bias to the result

    ; Activation code for Neuron 1
    maxss xmm0, dword [rel zero_val]   ; Apply ReLU activation function
    movss dword [rel h1], xmm0         ; Store the result in h1

```

This spells out, mathematically, the equation $y = w_1x_1 + w_2x_2 + b$. The ReLU activation in this case was done using the `maxss` instruction, as it efficiently checks if the result stored in `xmm0` is greater than zero, keeping it if it is, or forcing it to `0.0` if it isn't.

A similar approach was taken when calculating Neuron 2. However, for Neuron 3, the ReLU activation was removed so it could serve as the raw output layer of the model.

### 2. Computing the Error

The error calculation was straightforward. It simply takes the raw output computed by Neuron 3 at the end of the forward pass and subtracts it from the Target value initialized at the beginning of the epoch.

### 3. Backpropagation & The Chain Rule

Backpropagation for Neuron 1 looks like this:

```nasm
    ; --- Error for Neuron 1 --- 
    movss xmm5, xmm4                   ; Copy the Final Error
    mulss xmm5, dword [rel w31]        ; Hidden Error = Final Error * w31

    ; --- ReLU derivative for Neuron 1 ---
    movss xmm7, dword [rel h1]         ; Load Neuron 1's saved answer
    ucomiss xmm7, dword [rel zero_val] ; Compare it to 0.0
    ja .keep_er1                       ; If it's > 0, jump and keep the error
    movss xmm5, dword [rel zero_val]   ; Else, zero out the error!
.keep_er1:

```

This concept calculates the hidden error using the formula: `Hidden Error = Final Error * Weight`. This determines the exact amount of error contributed by Neuron 1.

Afterward, the **ReLU derivative** is performed. This step looks at the value stored in the neuron to see if it was turned off (`0.0`) during the forward pass. It essentially acts as the derivative: if the stored number is $\le 0$, the derivative is $0$ (so we wipe out the error). If it is $> 0$, the derivative is $1$ (so we keep the error to update the weights).

Finally, the weights are updated using the equation: `Weight = Weight + (Learning Rate * Hidden Error * Input)`. The biases are updated similarly: `Bias = Bias + (Learning Rate * Hidden Error)`.

This math is spelled out in the snippet below:

```nasm
    ; Update weights and bias for Neuron 1 

    ; w11 = w11 + (lr * error * input1)
    movss xmm2, dword [rel lr]
    mulss xmm2, xmm5 
    movss xmm3, dword [rsi]
    mulss xmm3, xmm2
    addss xmm3, dword [rel w11]
    movss dword [rel w11], xmm3 

    ; w12 = w12 + (lr * error * input2)
    movss xmm2, dword [rel lr]
    mulss xmm2, xmm5 
    movss xmm3, dword [rsi + 4]
    mulss xmm3, xmm2
    addss xmm3, dword [rel w12]
    movss dword [rel w12], xmm3 

    ; b1 = b1 + (lr * error)
    movss xmm2, dword [rel lr]
    mulss xmm2, xmm5 
    movss xmm3, dword [rel b1]
    addss xmm3, xmm2
    movss dword [rel b1], xmm3 

```

## The Results

After running the training loop for 50,000 epochs, the network successfully learned the non-linear XOR logic gate. Down to two decimal places, I was able to output:

```text
Result: 0.00 (Target: 0)
Result: 0.99 (Target: 1)
Result: 0.99 (Target: 1)
Result: 0.00 (Target: 0)

```