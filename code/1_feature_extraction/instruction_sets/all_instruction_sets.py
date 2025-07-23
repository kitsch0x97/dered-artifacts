# Auto-generated instruction sets
arith_set = {'DIVSS', 'PADDD', 'PADDSW', 'PMADDWD', 'INC', 'PADDW', 'SUBSS', 'ADDSD', 'NEG', 'DIV', 'ADDSS', 'MULSD', 'SUB', 'DEC', 'IDIV', 'MULSS', 'MUL', 'MAXSD', 'SBB', 'MINSD', 'ADC', 'SQRTSD', 'SUBSD', 'DIVSD', 'MINSS', 'ADD', 'IMUL'}
data_transfer_set = {'PUSH', 'MOVDQA', 'CVTTSS2SI', 'REPE', 'CVTPS2PD', 'CVTSI2SS', 'CVTSI2SD', 'STOSW', 'MOVUPS', 'MOVSXD', 'REP', 'MOVSW', 'STOSB', 'CQO', 'MOVDQU', 'MOVD', 'MOVZX', 'MOVAPD', 'MOVSB', 'CVTSD2SS', 'CDQE', 'BSWAP', 'MOVAPS', 'MOVSD', 'CWDE', 'POP', 'MOVQ', 'MOVSS', 'CVTSS2SD', 'CVTTSD2SI', 'MOVSX', 'CDQ', 'LEA', 'CVTPD2PS', 'MOV'}
cmp_set = {'PCMPEQW', 'CMPNLESD', 'CMP', 'UCOMISS', 'CMPLTSS', 'PCMPEQB', 'CMPLESD', 'PCMPEQD', 'TEST', 'UCOMISD', 'COMISD', 'COMISS'}
logic_set = {'ANDNPD', 'XOR', 'ANDNPS', 'ORPD', 'NOT', 'AND', 'BSR', 'PXOR', 'ORPS', 'ANDPS', 'XORPD', 'OR', 'ANDPD', 'XORPS'}
shift_set = {'SHR', 'ROR', 'SHL', 'PSLLD', 'SARX', 'SAR', 'ROL', 'SHRX', 'SHLX'}
unconditional_set = {'RETN', 'JMP', 'CALL'}
conditional_set = {'CMOVNS', 'JB', 'CMOVLE', 'JS', 'JA', 'JG', 'CMOVGE', 'JNP', 'JGE', 'CMOVL', 'JNZ', 'CMOVB', 'SETL', 'CMOVNB', 'CMOVA', 'CMOVZ', 'CMOVS', 'JNS', 'JP', 'SETNB', 'JE', 'JBE', 'JZ', 'JLE', 'JL', 'SETNZ', 'SETB', 'CMOVG', 'BT', 'CMOVBE', 'JNB', 'CMOVNZ', 'SETZ', 'SETLE', 'SETNLE', 'SETNL', 'SETNBE', 'SETNP', 'SETBE'}
memory_management_set = {'PREFETCHT1', 'PREFETCHT0'}
processor_state_set = {'XGETBV', 'CPUID'}
synchronization_set = {'PAUSE', 'LOCK', 'MFENCE', 'XCHG'}
vector_management_set = {'PSHUFD', 'VZEROUPPER', 'UNPCKLPD', 'PUNPCKLBW', 'PMOVMSKB', 'PSADBW', 'UNPCKLPS', 'PUNPCKLQDQ', 'PUNPCKLWD', 'PUNPCKHBW'}
unknown_set = {'UD2', 'UNKNOWN', 'TZCNT', 'NOP'}
