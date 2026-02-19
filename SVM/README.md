# DD2421/DD3431 Machine Learning - Lab 2: Support Vector Machines

## Overview
This repository contains the implementation and analysis for Lab 2: Support Vector Machines of the Machine Learning course at KTH Royal Institute of Technology. The lab focuses on building an SVM classifier by solving the dual optimization problem. It explores the mathematical formulation of the optimization task, the kernel trick, and the regularization effects of slack variables (C-value).

## Key Results

### 1. Dual Optimization & Support Vectors
* The SVM was implemented by minimizing the dual formulation of the optimization problem using the `minimize` function from the `scipy.optimize` package.
* Data points corresponding to non-zero $\alpha_{i}$ values were extracted to identify the support vectors.
* Because of floating-point arithmetic, a low threshold ($10^{-5}$) was used to accurately determine which $\alpha_{i}$ values are functionally non-zero.

### 2. Kernel Functions & Decision Boundaries
* The kernel trick was utilized to compute the scalar product of high-dimensional transformations implicitly, without excessive computational costs.
* **Linear Kernel**: Simply returns the scalar product between two points, resulting in a standard linear separation.
* **Polynomial Kernel**: Allows for curved decision boundaries (ellipses, parabolas) controlled by the degree parameter $p$.
* **Radial Basis Function (RBF)**: Uses the explicit Euclidean distance between points to generate highly smooth, complex boundaries. The smoothness is controlled by the $\sigma$ parameter.

### 3. Slack Variables & Bias-Variance Trade-off
* Slack variables ($\xi_{i}$) were introduced to allow a few datapoints to be misclassified, which is desirable when dealing with noisy, non-linearly separable data to achieve a substantially wider margin.
* The parameter $C$ dictates the relative importance of avoiding slack versus getting a wider margin.
* **High C**: Imposes a strict boundary with high penalties for slack, which can lead to overfitting noisy data.
* **Low C**: Allows for more slack, ensuring that individual noisy datapoints in strange locations do not overly influence the boundary, thereby improving generalization.

## Repository Structure
* `SVM.py`: Python script implementing the dual SVM, kernels, and `matplotlib` visualization routines.
* `svm-2018.pdf`: Lab instructions and theoretical background.

---

# DD2421/DD3431 机器学习 - Lab 2: 支持向量机 (SVM)

## 项目概览
本仓库包含瑞典皇家理工学院 (KTH) 机器学习课程 Lab 2: 支持向量机 的实现与分析。实验重点在于通过求解对偶优化问题来构建 SVM 分类器。探讨了优化任务的数学推导、核技巧 (Kernel Trick) 以及松弛变量（C值）的正则化效应。

## 核心实验结果

### 1. 对偶优化与支持向量
* 使用 `scipy.optimize` 库中的 `minimize` 函数来最小化对偶形式的优化问题。
* 通过提取非零的 $\alpha_{i}$ 值来定位支持向量。
* 由于浮点数精度的原因，采用了一个较低的阈值 ($10^{-5}$) 来准确判断哪些 $\alpha_{i}$ 值在实际意义上为非零值。

### 2. 核函数与决策边界
* 利用核技巧隐式计算高维空间转换后的内积，从而避免了极高的计算成本。
* **线性核 (Linear Kernel)**: 直接返回两点之间的标量乘积，产生线性的分类边界。
* **多项式核 (Polynomial Kernel)**: 允许生成弯曲的决策边界（如椭圆、抛物线），其复杂度由阶数参数 $p$ 控制。
* **径向基函数核 (RBF Kernel)**: 基于两点之间的欧几里得距离，能够生成极其平滑且复杂的边界。边界的平滑度由参数 $\sigma$ 控制。

### 3. 松弛变量与偏差-方差权衡
* 引入了松弛变量 ($\xi_{i}$) 以允许少量数据点被错误分类。在处理包含噪声且非线性可分的数据时，这种方法可以获得更宽的间隔。
* 参数 $C$ 决定了“避免松弛误差”与“获得更宽间隔”之间的相对重要性。
* **高 C 值**: 边界更严格，对误差的惩罚更大，容易对噪声数据产生过拟合。
* **低 C 值**: 允许更多的松弛误差，确保个别位置异常的噪声点不会过度影响决策边界，从而提高模型的泛化能力。

## 仓库结构
* `SVM.py`: 包含对偶 SVM 实现、核函数以及基于 `matplotlib` 的可视化 Python 脚本。
* `svm-2018.pdf`: 实验指导书及理论背景资料。
