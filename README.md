# HRF-HPL

基于 GRU 与超图融合的下一条 POI 推荐模型。

## 目录结构

```
HRF-HPL/
├── train.py                  # 训练入口
├── model.py                  # 模型定义
├── dataset.py                # 数据加载
├── dataset_config.py         # 数据集路径与规模配置
├── metrics.py                # 评估指标
├── utils.py                  # 工具函数
├── run_train_gowalla.bat     # Windows 一键训练
├── run_train_gowalla_bg.sh   # Linux 后台训练
└── tail_gowalla_log.sh       # 实时查看训练日志
```

训练输出默认写入 `logs/` 或 `logs_gowalla/`（由 `--save_dir` 指定），首次运行自动创建。

## 环境依赖

```bash
pip install torch numpy pyyaml
```

## 数据准备

数据集位于项目上级目录（与 `HRF-HPL` 同级）：

```
../ICASSP2024_ASTHL-main/datasets/{NYC|TKY|Gowalla}/
  train_poi_zero.txt
  test_poi_zero.txt
  {Dataset}_pois_coos_poi_zero.pkl
```

用户数、POI 数由 `dataset_config.py` 从 pickle 自动推断。

## 训练

```bash
cd HRF-HPL
python train.py --dataset Gowalla
```

支持的数据集：`NYC`、`TKY`、`Gowalla`。

常用参数：

```bash
python train.py --dataset Gowalla --batch_size 4 --epochs 200 --device cuda --save_dir logs_gowalla
```

Gowalla POI 较多（约 2 万+），首次加载会构建地理邻接矩阵，step2 可能较慢；Gowalla 下默认 `batch_size` 会自动从 8 调到 4。

### Windows

```bat
cd HRF-HPL
run_train_gowalla.bat
run_train_gowalla.bat --epochs 100 --batch_size 4
```

### Linux 后台运行

```bash
cd HRF-HPL
chmod +x run_train_gowalla_bg.sh tail_gowalla_log.sh
bash run_train_gowalla_bg.sh
bash run_train_gowalla_bg.sh --epochs 200 --batch_size 4 --device cuda

# 实时查看训练日志
bash tail_gowalla_log.sh

# 停止
kill $(cat logs_gowalla/train.pid)
```

## 日志说明

```
logs_gowalla/
├── runner_stdout.log      # 后台脚本标准输出
├── runner_stderr.log      # 后台脚本错误输出
├── train.pid              # 后台进程 PID
└── 20260615_142200/       # 单次训练 run
    ├── log_training.txt   # 训练指标（Recall / NDCG / loss）
    ├── Gowalla_args.yaml  # 超参数快照
    └── Gowalla.pt         # 最优模型权重
```
