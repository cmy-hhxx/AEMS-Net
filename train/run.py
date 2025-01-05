import subprocess
import sys
import time
import select
import fcntl
import os
import re


def run_training(model_name):
    command = [sys.executable, "train.py", f"--model_name={model_name}"]

    print(f"Starting training for {model_name}...")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
                               bufsize=1)

    # 设置非阻塞模式
    for pipe in [process.stdout, process.stderr]:
        flags = fcntl.fcntl(pipe.fileno(), fcntl.F_GETFL)
        fcntl.fcntl(pipe.fileno(), fcntl.F_SETFL, flags | os.O_NONBLOCK)

    progress_bar = ""
    while True:
        # 使用select来检查输出
        reads = [process.stdout.fileno(), process.stderr.fileno()]
        ret = select.select(reads, [], [], 0.1)

        for fd in ret[0]:
            if fd == process.stdout.fileno():
                read = process.stdout.read()
                if read:
                    lines = read.splitlines()
                    for line in lines:
                        if re.search(r'\d+%|\[=*\s*\]', line):  # 检查是否是进度条
                            progress_bar = line
                        else:
                            print(line)
            if fd == process.stderr.fileno():
                read = process.stderr.read()
                if read:
                    sys.stderr.write(read)
                    sys.stderr.flush()

        # 在每次循环结束时打印进度条
        if progress_bar:
            print(f"\r{progress_bar}", end="", flush=True)

        # 检查进程是否结束
        if process.poll() is not None:
            break

    # 确保进度条最后一次更新被打印出来
    if progress_bar:
        print(f"\r{progress_bar}")

    # 确保所有输出都被读取
    stdout, stderr = process.communicate()
    if stdout:
        print(stdout)
    if stderr:
        print(stderr)

    if process.returncode != 0:
        print(f"Error occurred while training {model_name}.")
    else:
        print(f"Training completed for {model_name}")


def main():
    # 首先训练UNet
    run_training("UNet")

    print("UNet training completed. Waiting for 60 seconds before starting UKan training...")
    time.sleep(60)  # 等待60秒

    # 然后训练AEMSN
    run_training("AEMSN")


if __name__ == "__main__":
    main()
