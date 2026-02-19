* # KTH DD2421/DD3431 Machine Learning - VT26

  Coursework and laboratory implementations for Machine Learning at KTH Royal Institute of Technology.

  ## 📂 Project Structure
  * `dectrees/`: Lab 1 - Decision Trees (ID3, Entropy & Pruning).
  * `svm/`: Lab 2 - Support Vector Machines (Dual Optimization, Kernels, Slack).
  * **Upcoming**: Lab 3 (Bayesian), Lab 4 (Boosting).

  ## 🧪 Lab 1 Summary: Decision Trees
  * **Entropy**: Calculated for MONK-1 (1.0), MONK-2 (0.957), and MONK-3 (0.999).
  * **Performance**: Full trees reached 0% training error but showed overfitting, especially on MONK-2 (30.79% test error).
  * **Pruning**: Optimized using Reduced Error Pruning with best training fractions at 0.6 (MONK-1) and 0.7 (MONK-3).

  ## 🧪 Lab 2 Summary: Support Vector Machines
  * **Dual Optimization**: Solved the SVM dual formulation using the `scipy.optimize.minimize` function to find the maximal-margin solution and extract support vectors.
  * **Kernels**: Evaluated Linear, Polynomial, and RBF kernels. Utilized the kernel trick to efficiently handle non-linear transformations and complex decision boundaries without excessive computational costs. 
  * **Slack Variables**: Analyzed the C-parameter's impact on the bias-variance trade-off. Explored how lowering the C-value allows for more slack and a wider margin, which is crucial for generalizing well on noisy, non-linearly separable data.

  ## ⚖️ Academic Integrity
  All files in this repository are intended only for study and course examination at KTH. Please do not use this repository for plagiarism, and do not redistribute the solutions in ways that violate KTH’s rules on collaboration and academic honesty.

  ## 🛠️ Tech Stack
  * Python 3 (NumPy, SciPy, Matplotlib)

  ---

  # KTH DD2421/DD3431 机器学习 - VT26

  本仓库包含瑞典皇家理工学院 (KTH) 机器学习课程的实验实现与分析。

  ## 📂 项目结构
  * `dectrees/`: Lab 1 - 决策树 (ID3 算法、熵分析与剪枝)。
  * `svm/`: Lab 2 - 支持向量机 (对偶优化、核函数、松弛变量)。
  * **待更新**: Lab 3 (贝叶斯)、Lab 4 (Boosting)。

  ## 🧪 Lab 1 实验总结：决策树
  * **熵 (Entropy)**: 训练集计算结果分别为 MONK-1 (1.0), MONK-2 (0.957), 以及 MONK-3 (0.999)。 
  * **模型表现**: 全树训练误差均达到 0%，但存在明显的过拟合现象，尤其在 MONK-2 上（测试误差 30.79%）。 
  * **剪枝 (Pruning)**: 使用减少误差剪枝 (Reduced Error Pruning) 进行优化，MONK-1 的最佳训练集比例约为 0.6，MONK-3 约为 0.7。

  ## 🧪 Lab 2 实验总结：支持向量机 (SVM)
  * **对偶优化 (Dual Optimization)**: 使用 `scipy.optimize.minimize` 函数求解 SVM 对偶形式，以寻找最大间隔解并提取出支持向量。
  * **核函数 (Kernels)**: 实现了线性核、多项式核与 RBF（径向基）核。利用“核技巧”直接计算标量乘积，从而在不增加极高计算成本的前提下高效处理非线性决策边界。
  * **松弛变量 (Slack Variables)**: 分析了 C 参数对偏差-方差权衡的影响。探讨了降低 C 值如何允许更多的松弛误差以获得更宽的间隔，这对于提升模型在非线性可分及噪声数据上的泛化能力至关重要。

  ## ⚖️ 学术诚信声明
  本仓库中的所有文件仅供 KTH 学习与课程考核参考。请勿将本项目用于剽窃，严禁以任何违反 KTH 协作与学术诚信条例的方式重新分发这些解决方案。

  ## 🛠️ 技术栈
  * Python 3 (NumPy, SciPy, Matplotlib)
