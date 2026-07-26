section .data

; XOR Gate inputs, targets
    inputs dd 0.0, 0.0 
           dd 0.0, 1.0
           dd 1.0, 0.0 
           dd 1.0, 1.0

    targets dd 0.0 
            dd 1.0 
            dd 1.0 
            dd 0.0

; Weights and biases
    w11 dd 0.5
    w12 dd -0.5 
    b1 dd 0.0 
    h1 dd 0.0

    w21 dd -0.5 
    w22 dd 0.5 
    b2 dd 0.0
    h2 dd 0.0

    w31 dd 0.5
    w32 dd 0.5
    b3 dd 0.0

    lr dd 0.01           ; Learning rate -- Controls how much we adjust the weights and bias during training

    zero_val dd 0.0 
    one_val dd 1.0

    ; Print definitions
    prefix db "Result: "
    pre_len equ $ - prefix
    dot db "."
    nl db 0x0A

section .bss
    buffer resb 16


section .text
    default rel 
    global _start


_start: 
    mov r8, 1000

.epoch_loop: 
    lea rbx, [rel inputs]
    lea r12, [rel targets]
    mov rcx, 0

.loop: 
    push rcx                     ; Save the current index
    push r8                      ; Save the epoch count

    mov rsi, rbx
    mov rdi, r12

    call forward_pass            ; Call the forward pass function to compute the outpu
    call errors                  ; Call the error function to compute the error
    call backpropagate           ; Call the backpropagation function to update weights and bias

    call print                   ; Call the print function to display the result

    add rbx, 8                   ; For every dd = 4 bytes
    add r12, 4

    pop r8                       ; Restore the epoch count
    pop rcx                      ; Restore the index

    inc rcx                      ; Increment the index
    cmp rcx, 4                   ; Check if we have processed all 4 input pairs
    jl .loop                    ; If not, repeat the loop

    dec r8                       ; Decrement the epoch count
    jnz .epoch_loop              ; If there are more epochs, repeat the loop

    mov rax, 60                 ; syscall: exit
    mov rdi, 0                  ; exit code 0
    syscall


forward_pass: 
; y = wn*xn + w(n+1)*x(n+1) + b

    ; Neuron 1 computations
    movss xmm0, dword [rel w11]        ; Load w11 into xmm0
    mulss xmm0, dword [rsi]            ; Multiply w11 with input

    movss xmm1, dword [rel w12]        ; Load w12 into xmm1
    mulss xmm1, dword [rsi + 4]        ; Load second input into xmm2
    
    addss xmm0, xmm1                      ; Add the two products together
    addss xmm0, dword [rel b1]            ; Add bias to the result

    ; Activation code for Neuron 1
    maxss xmm0, dword [rel zero_val]      ; Apply ReLU activation function
    movss dword [rel h1], xmm0
    
    ; Neuron 2 computations
    movss xmm0, dword [rel w21]        ; Load w11 into xmm0
    mulss xmm0, dword [rsi]            ; Multiply w11 with input

    movss xmm1, dword [rel w22]        ; Load w12 into xmm1
    mulss xmm1, dword [rsi + 4]        ; Load second input into xmm2
    
    addss xmm0, xmm1                      ; Add the two products together
    addss xmm0, dword [rel b2]            ; Add bias to the result

    ; Activation code for Neuron 2
    maxss xmm0, dword [rel zero_val]      ; Apply ReLU activation function
    movss dword [rel h2], xmm0

    ; Neuron 3 computations
    movss xmm0, dword [rel h1]            ; Load h1 into xmm0
    mulss xmm0, dword [rel w31]           ; Multiply h1 with w31

    movss xmm1, dword [rel h2]            ; Load h2 into xmm1
    mulss xmm1, dword [rel w32]           ; Multiply h2 with w32

    addss xmm0, xmm1                      ; Add the two products together
    addss xmm0, dword [rel b3]            ; Add bias to the result

    ret
 

errors: 
; Error = Target - Result
    movss xmm4, dword [rdi]
    subss xmm4, xmm0 
    ret

backpropagate: 

; Error for Neurons = Final Error*wnn 
; Hidden Error = Incoming Error * f'(x)

; Error for Neuron 1 
    movss xmm5, xmm4 
    mulss xmm5, dword [rel w31]

; Relu derivative for Neuron 1 
    movss xmm7, dword [rel h1]
    ucomiss xmm7, dword [rel zero_val]
    ja .keep_er1
    movss xmm5, dword [rel zero_val]
.keep_er1:
    
    ;Error for Neuron 2
    movss xmm6, xmm4 
    mulss xmm6, dword [rel w32] 

    ;Relu derivative for Neuron 2 
    movss xmm7, dword [rel h2]
    ucomiss xmm7, dword [rel zero_val]
    ja .keep_er2
    movss xmm6, dword [rel zero_val]
.keep_er2:


; Update weights and bias for Neuron 3 

; w31 = w31 + lr*error*input
   movss xmm2, dword [rel lr]
   mulss xmm2, xmm4 
   movss xmm3, dword [rel h1]
   mulss xmm3, xmm2
   addss xmm3, dword [rel w31]
   movss dword [rel w31], xmm3 

; w32 = w32 + lr*error*input
   movss xmm2, dword [rel lr]
   mulss xmm2, xmm4 
   movss xmm3, dword [rel h2]
   mulss xmm3, xmm2
   addss xmm3, dword [rel w32]
   movss dword [rel w32], xmm3 

; bias = bias + lr*error
   movss xmm2, dword [rel lr]
   mulss xmm2, xmm4 
   movss xmm3, dword [rel b3]
   addss xmm3, xmm2
   movss dword [rel b3], xmm3 
   

; Update weights and bias for Neuron 1 

; w11 = w11 + lr*error*input
   movss xmm2, dword [rel lr]
   mulss xmm2, xmm5 
   movss xmm3, dword [rsi]
   mulss xmm3, xmm2
   addss xmm3, dword [rel w11]
   movss dword [rel w11], xmm3 

; w12 = w12 + lr*error*input
   movss xmm2, dword [rel lr]
   mulss xmm2, xmm5 
   movss xmm3, dword [rsi + 4]
   mulss xmm3, xmm2
   addss xmm3, dword [rel w12]
   movss dword [rel w12], xmm3 

; bias = bias + lr*error
   movss xmm2, dword [rel lr]
   mulss xmm2, xmm5 
   movss xmm3, dword [rel b1]
   addss xmm3, xmm2
   movss dword [rel b1], xmm3 


; Update weights and bias for Neuron 2
   
; w21 = w21 + lr*error*input
   movss xmm2, dword [rel lr]
   mulss xmm2, xmm6 
   movss xmm3, dword [rsi]
   mulss xmm3, xmm2
   addss xmm3, dword [rel w21]
   movss dword [rel w21], xmm3 

; w22 = w22 + lr*error*input
   movss xmm2, dword [rel lr]
   mulss xmm2, xmm6 
   movss xmm3, dword [rsi + 4]
   mulss xmm3, xmm2
   addss xmm3, dword [rel w22]
   movss dword [rel w22], xmm3 

; bias = bias + lr*error
   movss xmm2, dword [rel lr]
   mulss xmm2, xmm6 
   movss xmm3, dword [rel b2]
   addss xmm3, xmm2
   movss dword [rel b2], xmm3 
   
   ret

print:                          ; Print function to display the result

    ; Print the prefix message
    mov rax, 1
    mov rdi, 1 
    lea rsi, [rel prefix] 
    mov rdx, pre_len 
    syscall 

    ; Extract the integer part
    cvttss2si rax, xmm0          ; Convert float in xmm0 to integer in rax
    add rax, '0'                 ; Convert integer to ASCII
    mov [rel buffer], al         ; Store ASCII character in buffer

    ; Print the integer part
    mov rax, 1
    mov rdi, 1 
    lea rsi, [rel buffer]
    mov rdx, 1
    syscall

    ; Print the dot
    mov rax, 1 
    mov rdi, 1                  ; Where?
    lea rsi, [rel dot]          ; What?
    mov rdx, 1                  ; How much?
    syscall

    ;Extract the fractional part
    cvttss2si rax, xmm0          ; Convert float in xmm0 to integer in rax
    cvtsi2ss xmm1, rax           ; Convert integer back to float in xmm1
    subss xmm0, xmm1             ; Subtract integer part from original float to get fractional part

    mov rax, 100
    cvtsi2ss xmm2, rax           ; Convert 100 to float in xmm2
    mulss xmm0, xmm2             ; Multiply fractional part by 100
    cvttss2si rax, xmm0          ; Convert fractional part to integer in rax

    ; Split the fractional part into two digits
    mov rcx, 10
    mov rdx, 0
    div rcx                      ; Divide by 10 to get tens and ones digits

    add rax, '0'                 ; Convert tens digit integer to ASCII
    add rdx, '0'                 ; Convert ones digit integer to ASCII

    mov [rel buffer], al         ; Store ASCII tens character in buffer
    mov [rel buffer+1], dl       ; Store ASCII ones character in buffer

    ; Print the fractional part
    mov rax, 1
    mov rdi, 1
    lea rsi, [rel buffer]
    mov rdx, 2
    syscall

    ; Print newline 
    mov rax, 1
    mov rdi, 1
    lea rsi, [rel nl]
    mov rdx, 1
    syscall
    ret