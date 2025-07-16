import os
import tqdm
import shutil
import subprocess
from multiprocessing import Process
from settings import IDA_PATH, FEATURE_EXTRACT_PATH

# 1. 用于处理二进制文件的函数
def processBinaryFiles(filePath, savePath):
    fileList = []
    tmp_extention = ['nam', 'til', 'id0', 'id1', 'id2', 'id3', 'id4', 'json', 'i64', 'a', "pkl"]
    files = os.walk(filePath)

    for path, dir_path, file_name in files:
        for file in file_name:
            if file.split(".")[-1] not in tmp_extention:
                fP = str(os.path.join(path, file))
                fileList.append(os.path.join(os.path.abspath('.'), fP))

    cfg_savePath = os.path.join(savePath, "cfg/")
    fcg_savePath = os.path.join(savePath, "fcg/")
    export_savePath = os.path.join(savePath, "export/")

    for path in [cfg_savePath, fcg_savePath, export_savePath]:
        os.makedirs(path, exist_ok=True)

    multiThreadProcessing(fileList, cfg_savePath, fcg_savePath, export_savePath, idapython=FEATURE_EXTRACT_PATH)

def multiThreadProcessing(fileList, cfg_savePath, fcg_savePath, export_savePath, idapython=""):
    process_num = min(30, len(fileList))
    p_list = []
    for i in range(process_num):
        files = fileList[int((i)/process_num*len(fileList)): int((i+1)/process_num*len(fileList))]
        p = Process(target=singleThreadProcessing, args=(files, cfg_savePath, fcg_savePath, export_savePath, idapython))
        p_list.append(p)
    for p in p_list:
        p.start()
    for p in p_list:
        p.join()

def singleThreadProcessing(filePaths, cfg_savePath, fcg_savePath, export_savePath, idapython):
    tf = tqdm.tqdm(filePaths)
    for filePath in tf:
        if not os.path.exists(os.path.join(os.path.basename(filePath) + "_timecost.json")):
            base_name = os.path.basename(filePath)
            cfg_save_name = os.path.join(cfg_savePath, base_name + "_cfg.json")
            fcg_save_name = os.path.join(fcg_savePath, base_name + "_fcg.pkl")
            export_save_name = os.path.join(export_savePath, base_name + "_export.json")

            (bpath, bbianry) = os.path.split(filePath)

            if not os.path.exists(os.path.join(bpath, bbianry+"_")):
                os.mkdir(os.path.join(bpath, bbianry+"_"))
                shutil.copy(filePath, os.path.join(bpath, bbianry+"_", bbianry))

            ida_cmd = 'TVHEADLESS=1 ' + IDA_PATH + ' -L/data/csj/DeRed/idalog.txt -c -A -B -S\'' + idapython + " " + cfg_save_name + " " + fcg_save_name + " " + export_save_name + " " + '\' ' + os.path.join(bpath, bbianry+"_", bbianry)

            try:
                process = subprocess.Popen(ida_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                process.communicate(timeout=1800)  # 设置超时时间为180秒
                s = process.returncode

            except subprocess.TimeoutExpired:
                process.kill()
                s = -1

            if s != 0:
                with open('error.txt', 'a') as file:
                    file.write(filePath + '\n')
                print("error: " + filePath)
            else:
                tf.set_description("[" + filePath.split("/")[-1] + "] Extract Success")