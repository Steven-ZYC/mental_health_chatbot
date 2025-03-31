
## 2. 简体中文版 README_zh_CN.md

```markdown:h%3A%5C%5B1%5DEdUHK%5COneDrive%20-%20The%20Education%20University%20of%20Hong%20Kong%5C%5B1%5DEdUHK%20Y1%20S2%5CINT2093%20%20n%20network%5Cgroup%20project%5Cbrandnew_test%5CREADME_zh_CN.md
# 心理健康支持聊天机器人

[English](README.md) | [简体中文](README_zh_CN.md) | [香港繁體中文](README_zh_HK.md)

一个基于人工智能的对话系统，通过情绪检测和响应式对话为用户提供心理健康支持。

## 核心功能

- **双模式情绪分析**：结合面部表情识别和文本情感分析
- **实时面部情绪检测**：使用计算机视觉技术检测面部表情中的情绪
- **文本情绪分析**：分析用户消息中的情感内容
- **危机指标检测**：识别潜在的心理健康危机信号
- **对话历史管理**：维护上下文以提供更相关的回应
- **两种操作模式**：
  - 用户模式：为普通用户提供简化界面
  - 开发者模式：提供实时指标和调试信息的高级界面

## 技术实现

### 1. 情绪识别模型 (EmotionDetector)

- **深度学习架构**：基于ResNet-18的卷积神经网络
  - 修改后的ResNet-18模型，适应灰度图像输入（单通道）
  - 输出层调整为7类情绪分类（happy, neutral, sad, surprise, angry, disgust, fear）
  
- **模型训练与优化**：
  - 使用PyTorch框架实现
  - 支持CPU和CUDA加速（自动检测可用设备）
  - 模型权重保存在`models/best_model.pth`

- **人脸检测**：
  - 使用OpenCV的Haar级联分类器进行人脸检测
  - 实时视频帧处理和情绪概率可视化

### 2. 语言处理系统 (LanguageProcessor)

- **文本情绪分析算法**：
  - 基于关键词匹配和情感强度分析
  - 情感强度评估基于感叹号、大写字母、字母重复和强度词汇
  
- **危机指标提取**：
  - 基于预定义关键词识别潜在危机情况
  - 支持多种危机类别识别（自伤、暴力、紧急情况等）

### 3. 响应生成系统 (ResponseGenerator)

- **上下文感知响应**：
  - 基于当前情绪状态和对话历史生成个性化回应
  - 情绪变化检测和相应的回应调整
  
- **危机响应机制**：
  - 优先级排序的危机响应策略
  - 集成支持资源和帮助热线信息

### 4. 多模态情绪融合

- **情绪状态验证算法**：
  - 结合视觉和文本情绪分析结果
  - 基于置信度的权重分配
  - 非中性情绪优先考虑策略

## 系统要求

- Python 3.8+
- PyTorch 2.0.1
- OpenCV 4.7.0
- CUDA兼容GPU（可选，用于更快的性能）

## 安装指南

1. 克隆仓库：
```bash
git clone <repository-url>
cd <repository-directory>