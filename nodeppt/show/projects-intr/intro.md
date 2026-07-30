title: 张逸为 - 项目经历介绍

speaker: 张逸为

css:
  - css/style.css



<slide class="bg-apple aligncenter">

<img src="https://uipv4.zywvvd.com:33030/HexoFiles/ppt/project-intr/media/avator.jpg" style="width:130px;height:130px;border-radius:50%;object-fit:cover;border:3px solid rgba(255,255,255,0.3);box-shadow:0 6px 24px rgba(0,0,0,0.45);margin-bottom:8px;">

---

# 个人经历介绍{.text-landing}

---

## 张逸为{.text-subtitle}

---

<font size=4 color=#BBB>

视觉算法 · 3D 重建 · 机器人规划 · 工业缺陷检测

---

8 年算法经验 · 4 家公司 · 7+ 项目

---

<font size=3 color=#999>

2026 年 6 月



<slide class="bg-white">

### 个人概要{.text-subtitle}

---

:::{.content-left}

<font size=4>

#### 基础信息

---

**电话**　18500585265

**邮箱**　zywvvd@mail.ustc.edu.cn

**博客**　www.zywvvd.com

**生日**　1991.10.12

**籍贯**　吉林吉林

:::

:::{.content-right}

<font size=4>

#### 教育背景

---

**2009.09 - 2013.06**

中国科学技术大学

信息科学技术学院 · 信息安全

**国家奖学金**

---

**2015.09 - 2018.06**

中国科学技术大学

信息科学技术学院 · 信息与通信工程

:::



<slide class="bg-white">

### 工作与项目经历{.text-subtitle}

---

<div style="border-left:3px solid #3D6EA5;padding-left:22px;margin-top:16px;font-size:19px;line-height:1.9;">

<div style="display:flex;gap:40px;margin-bottom:26px;">
<div style="min-width:195px;color:#3D6EA5;font-weight:700;">2018.07 - 2019.12</div>
<div>
<span style="font-size:23px;font-weight:700;color:#1a1a1a;">洛阳电子装备试验中心</span>
<span style="color:#777;">　视觉算法工程师</span>
<div style="color:#444;margin-top:4px;">深度学习对抗样本与图像隐写研究</div>
</div>
</div>

<div style="display:flex;gap:40px;margin-bottom:26px;">
<div style="min-width:195px;color:#3D6EA5;font-weight:700;">2020.01 - 2023.10</div>
<div>
<span style="font-size:23px;font-weight:700;color:#1a1a1a;">上海聚时科技</span>
<span style="color:#777;">　高级算法工程师</span>
<div style="color:#444;margin-top:4px;">JS6000 晶圆检测　/　晶圆 Review ADC　/　轴承·喷油嘴缺陷检测</div>
</div>
</div>

<div style="display:flex;gap:40px;margin-bottom:26px;">
<div style="min-width:195px;color:#3D6EA5;font-weight:700;">2023.12 - 2024.04</div>
<div>
<span style="font-size:23px;font-weight:700;color:#1a1a1a;">先导慧能</span>
<span style="color:#777;">　算法工程师</span>
<div style="color:#444;margin-top:4px;">iPhone 15 Plus 全机外观缺陷检测</div>
</div>
</div>

<div style="display:flex;gap:40px;">
<div style="min-width:195px;color:#3D6EA5;font-weight:700;">2024.05 - 至今</div>
<div>
<span style="font-size:23px;font-weight:700;color:#1a1a1a;">博珖机器人</span>
<span style="color:#777;">　视觉算法负责人 · 带 4 人团队</span>
<div style="color:#444;margin-top:4px;">BoEye 无人机施工系统　/　Bolight 机器人规划系统</div>
</div>
</div>

</div>

<div style="background:#f5f7fa;border-left:3px solid #3D6EA5;padding:12px 18px;margin-top:24px;color:#555;font-size:15px;line-height:1.7;">

<strong style="color:#3D6EA5;">技术主线</strong>　深度学习研究 → 工业缺陷检测 / 晶圆半导体 → 消费电子外观 → 3D 重建 / 机器人感知规划

</div>



<slide class="bg-white">

## 对抗样本与隐写分析研究

<font size=3 color=#888>洛阳电子装备试验中心　2018.07 - 2019.12 · 视觉算法工程师</font>

---

<font size=4>

**研究方向**　深度学习对抗样本 × 隐写分析神经网络

---

- 将**对抗样本**方法引入**隐写分析**场景，攻击多个隐写分析深度网络，显著提升其检测错误率
- 反向利用对抗扰动知识，构建更难被检测的隐写方案
- 参与深度学习**图像超分辨率**网络优化



<slide class="bg-white">

:::{.content-left}

## 轴承自动化缺陷检测

<font size=3 color=#888>上海聚时科技　2020.01 - 2021.08 · 高级算法工程师</font>

---

<font size=3>

**Hook 管线架构**　5 工位（面阵 / 线扫 / 内窥镜）协同检测

---

- **CV 算法**：RANSAC 椭圆拟合（5 环同心圆）、FFT 频域定位球 / 铆钉、小波变换 OCR
- **AI 模型**：ATSS 表面缺陷 / GFL 线扫旋转框 / YOLOX 密封件 / MobileNet 铆钉回归
- **合成数据**：ng_faker 生成器（OK 样本 + 缺陷实例库混合增强）
- **跨工位关联**：对比 1_2 面与 2_1 面铆钉尺寸偏差检测松动

:::

:::{.content-right}

<img src="https://uipv4.zywvvd.com:33030/HexoFiles/ppt/project-intr/media/bearing_defect.jpg" style="width:100%;height:560px;object-fit:cover;border-radius:8px;display:block;">

:::



<slide class="bg-white">

:::{.content-left}

## 喷油嘴多视角自动化检测

<font size=3 color=#888>上海聚时科技　2020.01 - 2021.08 · 高级算法工程师</font>

---

<font size=3>

**Observer DAG 框架**　5 工位 19 视角全覆盖

---

- **工位**：底部 / 线扫（4096×7500）/ 端面 / 肩部 / 内窥
- **六角镜**：6 折旋转对称 → 60° 步长模板匹配
- **FFT 相位对齐**：旋转工件极坐标 FFT → 主频相位 = 旋转角
- **异常检测**：PatchCore / FastFlow（内窥视角）
- **多视角 RGB 融合**：3 灰度视角合成彩色通道

:::

:::{.content-right}

<font size=2 color=#888>端面视角</font>
<img src="https://uipv4.zywvvd.com:33030/HexoFiles/ppt/project-intr/media/nozzle_tip_view.jpg" style="width:100%;height:175px;object-fit:cover;border-radius:4px;display:block;">

<font size=2 color=#888>底部视角</font>
<img src="https://uipv4.zywvvd.com:33030/HexoFiles/ppt/project-intr/media/nozzle_bottom_view.jpg" style="width:100%;height:175px;object-fit:cover;border-radius:4px;display:block;">

<font size=2 color=#888>内窥视角</font>
<img src="https://uipv4.zywvvd.com:33030/HexoFiles/ppt/project-intr/media/nozzle_endoscopy_2.jpg" style="width:100%;height:175px;object-fit:cover;border-radius:4px;display:block;">

:::



<slide class="bg-white">

:::{.content-left}

## 后道晶圆 Review 图像自动分类

<font size=3 color=#888>上海聚时科技　2021.09 - 2022.11 · 高级算法工程师</font>

---

<font size=3>

**Attention DAG 框架**（从喷油嘴 Observer DAG 演进）

---

- **三路并行**：Image / Template / ITP 图像-模板对齐
- **ITP 对齐**：4× 降采样粗搜 → 全分辨率精搜，7 层金字塔 + 0.5° 旋转步长
- **STPM 异常检测**：Teacher-Student 4 层加权特征差异
- **MAB 焊球缺失**：CNN 分割 + 匈牙利算法最优匹配
- **Matrix Ensemble**：分类器 + 检测器决策融合

支持 Klarf（KLA）/ SINF（SUSS）/ Camtek 多机台格式

**成果**　重要缺陷零漏检，降低过杀 **60%+**

:::

:::{.content-right}

<img src="https://uipv4.zywvvd.com:33030/HexoFiles/ppt/project-intr/media/adc_wafer_full.jpg" style="width:100%;height:560px;object-fit:cover;border-radius:8px;display:block;">

:::



<slide class="bg-white">

:::{.content-left}

## JS6000 高精度晶圆检测设备

<font size=3 color=#888>上海聚时科技　2022.12 - 2023.10 · 算法负责人（带 3 人团队）</font>

---

<font size=3>

**C++ / Halcon**　四大核心模块

---

- **RoutePlan**：1D 分解 + 2D 合并，Wafer→Shot→Die→ROI 四级坐标
- **2D Inspection**：金模比对（Min / Max / Sigma）+ Bin Code 8 类过滤
- **PMI**：9 项顺序检查（焊盘 / 探针印 / 偏位 / 形变 / 面积 / 触边 / 数量）
- **AutoFind**：NCC / Shape 双模态模板匹配 + 仿射变换

**指标**　亚微米级缺陷稳定检出

:::

:::{.content-right}

<img src="https://uipv4.zywvvd.com:33030/HexoFiles/ppt/project-intr/media/js6000_autofind.jpg" style="width:100%;max-height:580px;object-fit:contain;border-radius:6px;display:block;margin:auto;">

:::



<slide class="bg-white">

:::{.content-left}

## iPhone 15 Plus 全机外观缺陷检测

<font size=3 color=#888>先导慧能　2023.12 - 2024.04 · 算法工程师</font>

---

<font size=3>

**Cellink DAG 框架**　426 检测点位全覆盖

---

- **SimpleNet 异常检测**（CVPR 2023）：划伤 / 崩边 / 凹坑 / 色差 / 脏污
- **YOLO 目标检测**：自适应分块（1-4 块，640px tile + 160px overlap）
- **MobileNetV3-Small**：96×96 crop 二次分类去误报
- **深浅色自适应**：Logo 区域中值像素判断动态切模型
- **ONNX 部署**：CUDA + CPU 双后端推理

:::

:::{.content-right}

<img src="https://uipv4.zywvvd.com:33030/HexoFiles/ppt/project-intr/media/apple_golden.jpg" style="width:100%;height:560px;object-fit:cover;border-radius:8px;display:block;">

:::



<slide class="bg-white">

:::{.content-left}

## Bolight · 业务流程

<font size=3 color=#888>自主机器人规划与定位系统 · 光伏施工全流程</font>

---

<font size=3>

**项目简介**　面向光伏施工的自主机器人系统，覆盖起重机 / 机械臂 / 运输车的多智能体调度、运动规划、LiDAR 感知与视觉定位，基于 ROS 2 分层架构（规划 → 定位 → 感知 → 控制），让多设备在动态现场高效协同、精确作业。

**全流程**

- 任务下发 → 多车调度 → 单车路径规划 → 运动执行 → 视觉定位 / 安装
- 事件驱动重规划，动态场景持续可用
- 两阶段降级：全协调 → 超时退化局部规划

:::

:::{.content-right}

<video src="https://uipv4.zywvvd.com:33030/HexoFiles/ppt/project-intr/media/bolight_flow.mp4" data-autoplay loop muted playsinline controls style="max-width:100%;max-height:660px;width:auto;height:auto;border-radius:6px;display:block;margin:auto;"></video>

<font size=2 color=#888>业务流程演示</font>

:::



<slide class="bg-white">

:::{.content-left}

## Bolight · 单机运动学约束路径规划

<font size=3 color=#888>Reeds-Shepp 曲线 + 全局航向优化</font>

---

<font size=3>

**6 步管线**

- A* 宏观路径 → 道路 / 行内分段简化
- Reeds-Shepp 运动学曲线生成
- 2^N 翻转枚举全局航向优化（最小化原地转向）
- 位置 1cm / 角度 0.1° 连续性校验
- 五次多项式弧线平滑

:::

:::{.content-right}

<video src="https://uipv4.zywvvd.com:33030/HexoFiles/ppt/project-intr/media/bolight_demo.mp4" data-autoplay loop muted playsinline controls style="max-width:100%;max-height:792px;width:auto;height:auto;border-radius:6px;display:block;margin:auto;"></video>

<font size=2 color=#888>运动规划演示</font>

:::



<slide class="bg-white">

:::{.content-left}

## Bolight · 多智能体调度

<font size=3 color=#888>三层架构 + CBS 加权成本</font>

---

<font size=3>

**分层调度**

- 业务调度：卡车 5 状态生命周期
- SIPP 多智能体规划：时空联合搜索 + 安全间隔
- HA* 分层 A*：高层拓扑预规划 + 底层受限搜索
- CBS 加权成本取代 if-else 优先级
- 事件驱动多跳重规划

:::

:::{.content-right}

<video src="https://uipv4.zywvvd.com:33030/HexoFiles/ppt/project-intr/media/bolight_mapf.mp4" data-autoplay loop muted playsinline controls style="max-width:100%;max-height:792px;width:auto;height:auto;border-radius:6px;display:block;margin:auto;"></video>

<font size=2 color=#888>多车调度演示</font>

:::



<slide class="bg-white">

## BoEye 无人机施工采集系统

<font size=3 color=#888>无人机施工进度采集系统 · 博珖机器人　2024.05 - 至今 · 视觉算法负责人（带 4 人团队）</font>

---

:::{.content-left}

<font size=3>

**项目简介**　面向光伏电站建设的无人机施工进度采集系统：无人机自动航拍 → 3D 重建 → AI 分析 → 数字孪生，全自动输出施工进度、土方量、日报周报，实现现场无人值守的施工监测。

**无人机自动作业**

- DJI Cloud API + MQTT 接入 Dock 自动机场
- 远程任务下发 / 状态监控 / 断点续飞
- **10 种预设任务**（面 / 线 / 点 / 区扫描）

**分布式底座**

- **22 微服务**（17 算法 + 5 数据库）Docker 化
- ZMQ X-PUB/X-SUB 消息总线进程间解耦
- MySQL + MongoDB 分片 + Redis 锁 + MinIO

:::

:::{.content-right}

<img src="https://uipv4.zywvvd.com:33030/HexoFiles/ppt/project-intr/media/boeye_jiagou.png" style="width:100%;max-height:560px;object-fit:contain;border-radius:6px;display:block;">

<font size=2 color=#888>系统架构</font>

:::



<slide class="bg-white">

## BoEye · 3D 重建与数字孪生

<font size=3 color=#888>大规模影像 → 稠密模型 → 业务可读信息</font>

---

:::{.content-left}

<font size=3>

**3D 重建管线**

- ALIKED 深度特征（自写 CUDA kernel）→ COLMAP 稀疏 → OpenMVS 稠密
- DBSCAN 空间聚类切分大场景，子任务并行重建
- GPS 先验 BA、断点续建、623 组参数实验调优

**数字孪生**

- 16bit 高程图编码（亚厘米精度）
- 栅格差分对比两期 DEM 算土方挖填
- 50m 中值滤波 + Z-offset 校正

:::

:::{.content-right}

<div style="display:flex;gap:6px;">
<div style="flex:1;text-align:center;">
<img src="https://uipv4.zywvvd.com:33030/HexoFiles/ppt/project-intr/media/boeye_3d_ziyan.jpg" style="width:100%;max-height:480px;object-fit:contain;border-radius:6px;display:block;">
<font size=2 color=#888>自研管线</font>
</div>
<div style="flex:1;text-align:center;">
<img src="https://uipv4.zywvvd.com:33030/HexoFiles/ppt/project-intr/media/boeye_3d_dji.jpg" style="width:100%;max-height:480px;object-fit:contain;border-radius:6px;display:block;">
<font size=2 color=#888>DJI Terra</font>
</div>
</div>

:::



<slide class="bg-white">

:::{.content-left}

## BoEye · AI 语义分割

<font size=3 color=#888>道路 / 边界 / 建筑 / 组件等多类施工要素</font>

---

<font size=3>

**模型**

- DDRNet 语义分割，按类别训练多模型（道路 / 边界 / 沟槽 / 建筑 / 支架 / 组件）
- InferManager 统一加载、多模型协同推理

**部署**

- 导出 ONNX，ONNX Runtime GPU 推理
- 部署到现场工控机，产线实时运行

:::

:::{.content-right}

<video src="https://uipv4.zywvvd.com:33030/HexoFiles/ppt/project-intr/media/boeye_ai.mp4" data-autoplay loop muted playsinline controls style="max-width:100%;max-height:660px;width:auto;height:auto;border-radius:6px;display:block;margin:auto;"></video>

<font size=2 color=#888>分割识别演示</font>

:::



<slide class="bg-white">

:::{.content-left}

## BoEye · 全场正射识别

<font size=3 color=#888>正射图 + AI 识别叠加，施工要素一张图</font>

---

<font size=3>

**结果可视化**

- 全场正射图 + AI 分割结果叠加
- 道路 / 组件 / 支架等一张图总览

**业务报告**

- 日报 / 周报自动生成（多主题图表）
- 土方挖填量、施工进度等指标
- 打通"几何数据 → 管理决策"链路

:::

:::{.content-right}

<img src="https://uipv4.zywvvd.com:33030/HexoFiles/ppt/project-intr/media/boeye_panorama.jpg" style="max-width:100%;max-height:780px;width:auto;height:auto;object-fit:contain;display:block;margin:auto;border-radius:8px;">

:::



<slide class="bg-black aligncenter" image="https://uipv4.zywvvd.com:33030/HexoFiles/ppt/project-intr/media/boeye_result_local.jpg">

## BoEye · 局部识别结果{.text-shadow}

<font color=#FFF size=4 style="text-shadow:0 2px 10px rgba(0,0,0,.85);">

全景任一区域可追溯，组件 / 支架 / 桩基逐类标注

</font>



<slide class="bg-apple aligncenter">

### 技术主线与核心能力{.text-subtitle}

---

:::shadowbox

<font size=4>

**算法广度**：工业 2D / 晶圆半导体 / 消费电子 / 3D 重建 / 机器人感知规划

---

**工程深度**：从模块研发到团队带领，从算法设计到现场交付

---

**技术积累**：1093 篇原创技术博客 + 6 篇论文 + 10 项专利，持续学习与沉淀

---

**8 年 · 4 公司 · 7+ 项目 · 跨领域算法复用能力**

:::



<slide class="bg-apple aligncenter">

# 感谢聆听{.text-landing}

---

<font size=4 color=#BBB>

Thanks · 欢迎提问与交流
