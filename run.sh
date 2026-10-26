#!/usr/bin/env bash
# 实验2 一键运行脚本：打包 -> 上传输入 -> 运行全部任务 -> 下载结果
set -euo pipefail

# ---------- 环境配置（可按需修改） ----------
HADOOP_HOME="${HADOOP_HOME:-/home/legend/hadoop_installs/hadoop-3.5.0}"
MVN_HOME="${MVN_HOME:-/home/legend/apache-maven-3.9.9}"

HADOOP_BIN="${HADOOP_HOME:+$HADOOP_HOME/bin/}hadoop"
MVN_BIN="${MVN_HOME:+$MVN_HOME/bin/}mvn"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

JAR="target/lab2-1.0.jar"
TRAIN="train.csv"
TEST="test.csv"

HDFS_ROOT="/user/${USER:-$(whoami)}/lab2"
HDFS_TRAIN="$HDFS_ROOT/train.csv"
HDFS_TEST="$HDFS_ROOT/test.csv"

# ---------- 1. 编译打包 ----------
echo "==> 编译打包"
"$MVN_BIN" -q clean package -DskipTests

# ---------- 2. 准备 HDFS 输入 ----------
echo "==> 上传输入到 HDFS"
"$HADOOP_BIN" fs -mkdir -p "$HDFS_ROOT"
"$HADOOP_BIN" fs -put -f "$TRAIN" "$TEST" "$HDFS_ROOT/"

# ---------- 3. 清理旧输出 ----------
echo "==> 清理旧输出"
"$HADOOP_BIN" fs -rm -r -f \
    "$HDFS_ROOT/task1" "$HDFS_ROOT/task2" "$HDFS_ROOT/task2_avg" \
    "$HDFS_ROOT/task3" "$HDFS_ROOT/task4" \
    "$HDFS_ROOT/task5_model" "$HDFS_ROOT/task5"

# ---------- 4. 依次运行 5 个任务 ----------
echo "==> 任务1：各开户机构客户数量及资产统计"
"$HADOOP_BIN" jar "$JAR" lab2.Task1 "$HDFS_TRAIN" "$HDFS_ROOT/task1"

echo "==> 任务2：平均值分析"
"$HADOOP_BIN" jar "$JAR" lab2.Task2 "$HDFS_ROOT/task1" "$HDFS_TRAIN" "$HDFS_ROOT/task2"

echo "==> 任务3：客户总资产四分位数"
"$HADOOP_BIN" jar "$JAR" lab2.Task3 "$HDFS_TRAIN" "$HDFS_ROOT/task3"

echo "==> 任务4：客户资产分层与资产结构统计"
"$HADOOP_BIN" jar "$JAR" lab2.Task4 "$HDFS_TRAIN" "$HDFS_ROOT/task3" "$HDFS_ROOT/task4"

echo "==> 任务5：客户信贷风险预测（高斯朴素贝叶斯）"
"$HADOOP_BIN" jar "$JAR" lab2.Task5 "$HDFS_TRAIN" "$HDFS_TEST" "$HDFS_ROOT/task5_model" "$HDFS_ROOT/task5"

# ---------- 5. 下载结果到本地 output/ ----------
echo "==> 下载结果到 output/"
mkdir -p output/task1 output/task2 output/task3 output/task4 output/task5
"$HADOOP_BIN" fs -get "$HDFS_ROOT/task1/part-r-00000" output/task1/part-r-00000
"$HADOOP_BIN" fs -get "$HDFS_ROOT/task2/part-r-00000" output/task2/part-r-00000
"$HADOOP_BIN" fs -get "$HDFS_ROOT/task3/part-r-00000" output/task3/part-r-00000
"$HADOOP_BIN" fs -get "$HDFS_ROOT/task4/part-r-00000" output/task4/part-r-00000
"$HADOOP_BIN" fs -get "$HDFS_ROOT/task5/part-m-00000" output/task5/part-m-00000

echo "==> 完成"
echo "任务3四分位数: $(cat output/task3/part-r-00000)"
echo "任务5预测分布: 0=$(awk -F'\t' '$2==0' output/task5/part-m-00000 | wc -l)  1=$(awk -F'\t' '$2==1' output/task5/part-m-00000 | wc -l)"
