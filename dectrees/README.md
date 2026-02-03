# DD2421/DD3431 Machine Learning - Lab 1: Decision Trees

## Overview
This repository contains the implementation and analysis for **Lab 1: Decision Trees** of the Machine Learning course at KTH Royal Institute of Technology. The lab focuses on the **ID3 algorithm** and its performance on the **MONK datasets**, exploring concepts such as Entropy, Information Gain, Overfitting, and Reduced Error Pruning.

## Key Results

### 1. Entropy & Information Gain
The entropy for each training set was calculated to measure uncertainty:
* **MONK-1**: 1.000000
* **MONK-2**: 0.957117
* **MONK-3**: 0.999806

At the root node, the attribute with the highest **Information Gain** was chosen for splitting:
* **MONK-1 & MONK-2**: Attribute $a_5$
* **MONK-3**: Attribute $a_2$

### 2. Full Tree Performance
All full decision trees achieved **0.00% training error** due to high model capacity. However, test errors revealed different levels of generalization:

| Dataset | Test Error (Full Tree) | Complexity Analysis |
| :--- | :--- | :--- |
| **MONK-1** | 0.171296 | Relatively simple rule, intermediate error. |
| **MONK-2** | 0.307870 | Most difficult; greedy splits struggle with counting rules. |
| **MONK-3** | 0.055556 | Contains noise; best performance but prone to fitting noise. |

### 3. Pruning & Bias-Variance Trade-off
Reduced Error Pruning was applied to regularize the trees. By replacing subtrees with leaves, we reduced **variance** at the cost of a slight increase in **bias**.

* **Optimal Training Fraction**: Around **0.6** for MONK-1 and **0.7** for MONK-3.
* **Observation**: Small fractions underfit due to lack of data, while large fractions yield unstable pruning due to insufficient validation data.

## Repository Structure
* `python/`: Python scripts including `dtree.py` and dataset loaders.
* `DD2421_Lab1_report.pdf`: Detailed experimental report and analysis.
* `dectrees-py.pdf`: Lab instructions and background information.



# DD2421/DD3431 机器学习 - Lab 1: 决策树

## 项目概览
本仓库包含瑞典皇家理工学院 (KTH) 机器学习课程 **Lab 1: 决策树** 的实现与分析。实验重点在于 **ID3 算法** 及其在 **MONK 数据集** 上的表现，探讨了熵、信息增益、过拟合以及减少误差剪枝 (Reduced Error Pruning) 等核心概念 。

## 核心实验结果

### 1. 熵与信息增益
计算了每个训练集的熵值以衡量不确定性：
* **MONK-1**: 1.000000 
* **MONK-2**: 0.957117 
* **MONK-3**: 0.999806 

在根节点处，根据最高**信息增益**原则选择分裂属性：
* **MONK-1 & MONK-2**: 选择属性 $a_5$ 
* **MONK-3**: 选择属性 $a_2$ 

### 2. 全树表现与分析
由于 ID3 算法强大的模型容量，所有全决策树的 **训练误差均为 0.00%**。但测试误差反映了不同的泛化能力 ：

| 数据集     | 测试误差 (全树) | 复杂度分析                                   |
| :--------- | :-------------- | :------------------------------------------- |
| **MONK-1** | 0.171296        | 底层规则相对简单，误差中等 。                |
| **MONK-2** | 0.307870        | 最难学习；贪心分裂难以捕捉全局计数规则 。    |
| **MONK-3** | 0.055556        | 包含噪声；表现最好但仍存在过拟合噪声的倾向。 |

### 3. 剪枝与偏差-方差权衡
通过减少误差剪枝进行正则化处理。通过降低**方差**（降低对噪声的敏感度）来提升稳定性，虽然可能会略微增加**偏差** 。

* **最佳训练集比例**: MONK-1 约为 **0.6**，MONK-3 约为 **0.7** 。
* **结论**: 训练比例过小时容易因数据不足导致欠拟合；比例过大时则因验证集过小导致剪枝决策不稳定。

## 仓库结构
* `python/`: 包含 `dtree.py` 及数据集加载器的 Python 脚本。
* `DD2421_Lab1_report.pdf`: 详细的实验报告与分析文档。
* `dectrees-py.pdf`: 实验指导书及背景资料。
