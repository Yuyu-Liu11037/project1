#!/usr/bin/env bash

# 批量运行 diagnosis_prediction.py 不同参数组合
# 所有输出都会追加写入同一个日志文件
# 即使中途某个命令报错，脚本也会继续执行后面的命令

set -u  # 使用未定义变量时报错退出；不使用 set -e，以便出错后继续下一个命令

###############################
# 配置区：根据需要修改下面参数
###############################

# 要尝试的随机种子列表
SEEDS=(24 44)

# 要尝试的 arch 列表
ARCHES=(
  "L3_W256"
  "L3_W384"
  "L3_W320"
  "L3_W448"
  "L3_W512"
  "L3_W576"
)

# Python 程序绝对路径
PY_SCRIPT="/data/yuyu/project1/diagnosis_prediction.py"

# 日志文件路径（绝对路径或相对路径均可）
LOG_FILE="/data/yuyu/project1/diagnosis_prediction_runs.log"

###############################
# 运行区：一般不需要修改
###############################

echo "========================================" >> "${LOG_FILE}"
echo "$(date '+%Y-%m-%d %H:%M:%S')  Start batch runs" >> "${LOG_FILE}"
echo "Log file: ${LOG_FILE}" >> "${LOG_FILE}"

for seed in "${SEEDS[@]}"; do
  for arch in "${ARCHES[@]}"; do
    echo "----------------------------------------" | tee -a "${LOG_FILE}"
    echo "$(date '+%Y-%m-%d %H:%M:%S')  Running: seed=${seed}, arch=${arch}" | tee -a "${LOG_FILE}"

    # 实际运行命令
    python "${PY_SCRIPT}" \
      --seed "${seed}" \
      --arch "${arch}" \
      >> "${LOG_FILE}" 2>&1

    # 记录每个组合的退出状态，但不因此中断后续运行
    status=$?
    if [ ${status} -ne 0 ]; then
      echo "$(date '+%Y-%m-%d %H:%M:%S')  Command FAILED with status ${status} (seed=${seed}, arch=${arch})" | tee -a "${LOG_FILE}"
    else
      echo "$(date '+%Y-%m-%d %H:%M:%S')  Command SUCCEEDED (seed=${seed}, arch=${arch})" | tee -a "${LOG_FILE}"
    fi
  done
done

echo "========================================" >> "${LOG_FILE}"
echo "$(date '+%Y-%m-%d %H:%M:%S')  All runs finished" >> "${LOG_FILE}"


