import os
import pdb
import json
from utils.instructions_deepseek_api import promptGeneration, answerGeneration
from instruction_sets.all_instruction_sets import arith_set, data_transfer_set, cmp_set, logic_set, shift_set, unconditional_set, conditional_set, memory_management_set, processor_state_set, synchronization_set, vector_management_set, unknown_set


def save_instruction_sets():
    module_path = os.path.abspath("xxx/DeRed/code/1_feature_extraction/instruction_sets/all_instruction_sets.py")

    file_content = f"""# Auto-generated instruction sets
arith_set = {repr(arith_set)}
data_transfer_set = {repr(data_transfer_set)}
cmp_set = {repr(cmp_set)}
logic_set = {repr(logic_set)}
shift_set = {repr(shift_set)}
unconditional_set = {repr(unconditional_set)}
conditional_set = {repr(conditional_set)}
memory_management_set = {repr(memory_management_set)}
processor_state_set = {repr(processor_state_set)}
synchronization_set = {repr(synchronization_set)}
vector_management_set = {repr(vector_management_set)}
unknown_set = {repr(unknown_set)}
"""
    try:
        with open(module_path, 'w', encoding='utf-8') as f:
            f.write(file_content)
        print(f"\nSuccessfully updated {module_path}")
    except Exception as e:
        print(f"Error saving instruction sets: {str(e)}")
        raise

def readInstructions(instruction_path):
    all_instructions = set()
    for filename in os.listdir(instruction_path):
        if filename.endswith('.json'):
            file_path = os.path.join(instruction_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for key in data:
                    instructions_dict = data[key]['basic_blocks']
                    for instructions in instructions_dict:
                        instructions = instructions['instructions']
                        instructions = [instr.upper() for instr in instructions]
                        all_instructions.update(instructions)
    return all_instructions


def mappingTableGeneration(target_path, candidate_path):
    target_instructions = readInstructions(target_path)
    candidate_instructions = readInstructions(candidate_path)
    all_instructions = target_instructions | candidate_instructions
    exist_instructions = arith_set | data_transfer_set | cmp_set | logic_set | shift_set | unconditional_set | conditional_set | memory_management_set | processor_state_set | synchronization_set | vector_management_set | unknown_set

    instruction_sets = {
        "arith_set": arith_set, "data_transfer_set": data_transfer_set, "cmp_set": cmp_set, "logic_set": logic_set,
        "shift_set": shift_set, "unconditional_set": unconditional_set, "conditional_set": conditional_set,
        "memory_management_set": memory_management_set, "processor_state_set": processor_state_set, "synchronization_set": synchronization_set,
        "vector_management_set": vector_management_set, "unknown_set": unknown_set
    }

    missing_instructions = all_instructions - exist_instructions
    for instruction in missing_instructions:
        prompt = promptGeneration(instruction)
        answer = answerGeneration(prompt)
        if answer in instruction_sets:
            if instruction not in instruction_sets[answer]:
                instruction_sets[answer].add(instruction)
                save_instruction_sets()