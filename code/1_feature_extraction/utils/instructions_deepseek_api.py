import pdb
from openai import OpenAI

def answerGeneration(prompt):
    # client = OpenAI(api_key="sk-xxx", base_url="https://api.deepseek.com")
    client = OpenAI(api_key="sk-xxx", base_url="https://api.tu-zi.com/v1")

    max_retries = 3  
    retry_delay = 2  

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are an expert of binary analysis"},
                    {"role": "user", "content": prompt},
                ],
                stream=False
            )
            return response.choices[0].message.content.strip()
        except APIError as e:
            print(f"API request failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise
        except Exception as e:
            print(f"An unknown error occurred: {e}")
            raise


def promptGeneration(instruction):
    prompt = f"""
You are an instruction set classification expert. Your task is to map a given opcode to one of the eleven predefined categories based on its functionality. The classification rules are as follows:

1. **arith_set**: Arithmetic instructions, including addition, subtraction, multiplication, division, etc., but excluding data type conversions. Examples: ADD, SUB, MUL, DIV, NEG, SQRTSD.
2. **data_transfer_set**: Data transfer instructions, including data movement, stack operations, input/output, and data type conversions (e.g., float-to-int, sign extension). Examples: MOV, PUSH, POP, MOVZX, MOVSX, CWDE, MOVDQA, MOVUPS, REPNE.
3. **cmp_set**: Comparison instructions used to perform comparisons and set conditional flags. Includes both scalar and vector comparisons. Examples: CMP, TEST, PCMPGTD.
4. **logic_set**: Logical operations, including AND, OR, NOT, XOR, and other bitwise operations. Excludes bit-test instructions that set condition flags. Examples: AND, OR, NOT, XOR, ANDPD.
5. **shift_set**: Shift and rotate instructions, including logical shifts, arithmetic shifts, and rotates. Examples: SHL, SHR, ROL, ROR, PSLLW.
6. **unconditional_set**: Unconditional jump/call instructions, including direct jumps, procedure calls, and returns. Examples: JMP, CALL, RET.
7. **conditional_set**: Conditional jump and conditional move instructions, which depend on flags (e.g., equal, not equal), and bit-test instructions that set condition flags. Examples: JE, JNE, CMOVcc, BT, BTS.
8. **memory_management_set**: Memory management instructions for cache prefetching and memory barriers. Examples: PREFETCHT1, CLFLUSH.
9. **processor_state_set**: Processor state instructions for querying or managing processor information and state. Examples: XGETBV, XSAVE, CPUID.
10. **synchronization_set**: Synchronization instructions used for atomic operations, multithread synchronization, and busy-waiting. Examples: LOCK, CMPXCHG, XCHG, PAUSE, MFENCE.
11. **vector_management_set**: Vector management instructions used for managing vector register state (e.g., shuffle, pack, zeroing). Examples: VZEROUPPER, VPSHUFB, PUNPCKHBW.

Task Requirements:
- Given a single opcode as input (e.g., "ADD"), output the name of the corresponding category (e.g., `arith_set`).
- If the opcode does not belong to any of the above categories, output `unknown_set`.
- Strictly follow the input/output format. Do not add any extra explanation.

Examples:
Input: MOV    → Output: data_transfer_set  
Input: JE     → Output: conditional_set  
Input: ADD    → Output: arith_set  
Input: NOP    → Output: unknown_set  
Input: LOCK   → Output: synchronization_set  
Input: PAUSE  → Output: synchronization_set  
Input: MOVDQA → Output: data_transfer_set  
Input: PCMPGTD→ Output: cmp_set

Now start classifying:  
Input: {instruction}  
Output:
"""

    return prompt
