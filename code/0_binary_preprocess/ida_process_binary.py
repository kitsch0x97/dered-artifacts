# -*- coding: utf-8 -*-
from settings import *
import sys, os
sys.path.insert(0, PACKAGE_PATH)

import re
import idc
import json
from idc import *
from idaapi import *
import networkx as nx
from ida_gdl import *
# from getAcfg import *
from idautils import *
from ida_name import *
from ida_entry import *
from ida_funcs import *
from ida_segment import *
from ida_nalt import *
from collections import defaultdict

def extract_control_flow_graph(func_addr):
    """提取给定函数的控制流图（CFG）信息，包括操作码和操作数列表"""
    cfg = ida_gdl.FlowChart(ida_funcs.get_func(func_addr))

    basic_blocks = []
    for block in cfg:
        opcodes = []
        operands = []
        addr = block.startEA

        while addr < block.endEA:
            disasm = idc.GetDisasm(addr)
            opcode = extract_opcode(disasm)
            operand = extract_operands(disasm)

            opcodes.append(opcode)
            operands.append(operand if operand else "")

            addr = idc.NextHead(addr)

        block_info = {
            "start": hex(block.startEA),
            "end": hex(block.endEA),
            "successors": [hex(succ_block.startEA) for succ_block in block.succs()],
            "instructions": opcodes,
            "operands": operands
        }
        basic_blocks.append(block_info)

    return basic_blocks



def extract_opcode(disasm):
    """提取反汇编指令中的操作符"""
    match = re.match(r'\S+', disasm)
    return match.group(0) if match else ""


def extract_operands(disasm):
    """从反汇编指令中提取操作数列表，仅保留地址表达式中加号前的寄存器名"""
    match = re.match(r'\S+\s+(.*)', disasm)
    if not match:
        return []

    # 去除注释
    operands_str = match.group(1).split(';')[0]

    raw_operands = [op.strip() for op in operands_str.split(',') if op.strip()]
    processed_operands = []

    for op in raw_operands:
        # 匹配类似 [rdi+18h] 或 [rsp+98h+var_78]
        mem_match = re.match(r'\[\s*([a-zA-Z0-9_]+)', op)
        if mem_match:
            base_reg = mem_match.group(1)
            processed_operands.append(base_reg)
        else:
            processed_operands.append(op)

    return processed_operands

def extract_features():
    cfg_savePath = idc.ARGV[1]
    fcg_savePath = idc.ARGV[2]
    exprot_savePath = idc.ARGV[3]

    func_addr_dict = {}  # 存储符合条件的函数名与地址
    functions_info = {}
    exported_functions_info = {}
    callees = defaultdict(set)
    g = nx.DiGraph()

    text_seg = ida_segment.get_segm_by_name(".text")
    if not text_seg:
        print("`.text` segment not found.")
        return

    text_seg_start = text_seg.startEA
    text_seg_end = text_seg.endEA

    # 第一步：筛选符合条件的函数到func_addr_dict
    for func_addr in Functions():
        if text_seg_start <= func_addr < text_seg_end:
            func_name = ida_funcs.get_func_name(func_addr)
            if func_name == 'main':
                continue

            # 初步提取CFG进行过滤
            basic_blocks = extract_control_flow_graph(func_addr)
            if len(basic_blocks) <= 5 or sum(len(b['instructions']) for b in basic_blocks) <= 10:
                continue

            func_addr_dict[func_name] = func_addr

    # 第二步：收集调用关系构建FCG
    for func_name, func_addr in func_addr_dict.items():
        # 遍历所有调用该函数的位置
        for ref_ea in CodeRefsTo(func_addr, 0):
            if text_seg_start <= ref_ea < text_seg_end:
                caller_name = GetFunctionName(ref_ea)
                # 仅记录符合条件的调用者
                if caller_name in func_addr_dict:
                    callees[caller_name].add(func_name)

    # 构建函数调用图，只添加存在边的函数
    for caller, callees_set in callees.items():
        for callee in callees_set:
            g.add_edge(caller, callee)

    # 第三步：生成详细的CFG信息，只处理在FCG中存在的函数
    for func_name in g.nodes:
        func_addr = func_addr_dict[func_name]
        functions_info[func_name] = {
            "address": hex(func_addr),
            "basic_blocks": extract_control_flow_graph(func_addr)
        }

    # 处理导出函数
    entry_qty = get_entry_qty()
    for i in range(entry_qty):
        func_addr = get_entry(get_entry_ordinal(i))
        func_name = get_ea_name(func_addr)
        exported_functions_info[func_name] = {"address": hex(func_addr)}

    # 保存结果
    with open(cfg_savePath, "w") as f:
        json.dump(functions_info, f, indent=4)

    nx.write_gpickle(g, fcg_savePath)

    with open(exprot_savePath, "w") as f:
        json.dump(exported_functions_info, f, indent=4)

Wait()
extract_features()
Exit(0)