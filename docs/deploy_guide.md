# 智算天璇（OmniArch）本地部署与使用指南

## 一、这是什么

智算天璇（英文名 OmniArch）是一个偏微分方程基础模型。给它一段仿真的前几帧，它自己往后推演后续的场。一套权重同时处理一维、二维、三维问题，不需要告诉它方程和参数。

这里部署的是用 PyTorch 重新实现、在 PDEBench 上重新训练的模型（代码仓库 omniarch_pt，当前版本 v0.2）。它有三档权重：

| 权重 | 参数量 | 平均误差 | 适合的机器 |
|---|---|---|---|
| omniarch-base.pt | 3.08亿 | 0.159 | 默认推荐，有无显卡都能跑 |
| omniarch-mini.pt | 7700万 | 0.168 | 显存或内存较小的机器 |
| omniarch-tiny.pt | 1400万 | 0.212 | 笔记本、老旧机器 |

部署好之后有四种用法：网页演示、命令行、HTTP 接口、Python 接口。

## 二、它怎么"解"方程，能解哪些问题

### （一）预测求解是怎么回事

传统求解器从初值出发，每一小步按方程算出下一步。结果精确，但三维问题动辄要算几小时。

OmniArch 走的是另一条路。它在训练中看过大量方程的解怎样随时间演化。拿到一段新的仿真，它先从给定的前几帧里看出这是怎样的物理过程，比如流动多快、扩散多强、激波往哪走，再一帧一帧往后推。这和大语言模型读完前文接着往下写是同一个道理，只是它续写的是流场。

由此有两点要记住：

- **它是接着推，不是从零解方程。**输入至少要有几帧真实数据，来自求解器、实验或观测都可以，建议 10 帧。
- **它给的是快速预估，不是认证精度的解。**模型不保证质量、能量严格守恒，误差随推演步数累积。适合方案筛选、实时预览、给求解器提供初值，不适合直接替代最终的校核计算。

### （二）格式对了就能算，但不一定算得准

理论上，只要满足以下条件，任何一维、二维、三维、随时间演化的多物理场问题都能输入模型，并得到一个预测：

- 放在等间距的规则网格上；
- 物理量不超过 8 个，按固定顺序排列；
- 至少给出 1 帧，建议 10 帧。

但格式对不等于结果准。准不准取决于新问题和训练中见过的物理有多接近：

| 情况 | 能不能用 |
|---|---|
| 训练中见过的方程类型，参数和分辨率接近 | 可以直接用，误差见下面第一张表 |
| 没见过、但物理机制相近的问题 | 可能可用，必须先按第六节验证 |
| 物理机制差别大的问题，如电磁场、多相流、化学反应网络、固体力学 | 不建议直接用 |

训练中见过的方程类型和测试误差（相对误差 nRMSE，从第 10 帧往后一直推到最后一帧，越低越好；对照是"一直复制最后一帧"）：

| 问题类别 | 维度 | base | mini | tiny | 对照 |
|---|---|---|---|---|---|
| 平流 | 1D | 0.170 | 0.200 | 0.354 | 1.001 |
| Burgers | 1D | 0.146 | 0.145 | 0.172 | 0.335 |
| 可压缩流 | 1D | 0.139 | 0.146 | 0.249 | 0.453 |
| 扩散吸附 | 1D | 0.039 | 0.032 | 0.028 | 0.157 |
| 反应扩散 | 1D | 0.043 | 0.043 | 0.047 | 0.042 |
| 可压缩 NS | 2D | 0.236 | 0.250 | 0.326 | 0.648 |
| 扩散反应 | 2D | 0.142 | 0.156 | 0.164 | 0.496 |
| 浅水 | 2D | 0.004 | 0.004 | 0.005 | 0.110 |
| 可压缩 NS 和湍流 | 3D | 0.518 | 0.535 | 0.565 | 0.886 |
| **平均** | | **0.159** | **0.168** | **0.212** | **0.459** |

一维反应扩散的轨迹几帧之内就趋于均匀，复制最后一帧已经接近正确答案，所以模型反而略差。三维是最弱的一项，训练数据最少。

没见过的问题（零样本）在四个例子里三个优于对照、一个不如对照：

| 零样本问题 | OmniArch base | 对照 |
|---|---|---|
| 一维 Sod 激波管 | 0.011 | 0.018 |
| 二维 Orszag–Tang 涡 | 0.234 | 0.434 |
| 二维 Kelvin–Helmholtz 不稳定性 | 0.358 | 0.301 |
| 三维湍流，256³ 降采样到 32³ | 0.623 | 1.003 |

### （三）当前版本处理不了的情况

- 不规则网格、复杂几何外形：只能先插值到规则网格，边界附近误差会很大；
- 稳态问题：没有时间演化，比如达西渗流的压力分布；
- 很长时间的推演：误差逐步累积。目前只验证到一维往后推 31 步、二维和三维推 11 步，更长的推演没有验证过；
- 物理量超过 8 个。

网页演示就是上面这一套流程的可视化。左侧每个算例都是一份按格式整理好的数据，标着"训练分布内"或"零样本外推"。建议部署后先逐个点一遍，对模型能做到什么程度有个直观印象。

## 三、机器要求

| 项目 | 最低 | 推荐 |
|---|---|---|
| 操作系统 | Linux、Windows 10/11、macOS | Ubuntu 22.04 |
| Python | 3.9 | 3.10 到 3.12 |
| 内存 | 4GB | 16GB |
| 显卡 | 不需要 | NVIDIA 显卡，显存 8GB 以上，驱动支持 CUDA 12 |
| 磁盘 | 3GB（含 PyTorch） | 10GB |

在 8 核 CPU 上实测（自带算例）：

| 算例 | omniarch-base | omniarch-tiny |
|---|---|---|
| 一维，256 点，往后推 31 步 | 1.8 秒 | 0.1 秒 |
| 二维，64×64，4 个物理量，推 11 步 | 2.2 秒 | 0.5 秒 |
| 三维，32³，5 个物理量，推 11 步 | 10.5 秒 | 4.6 秒 |

mini 在 CPU 上的速度和 base 接近，因为耗时主要在解码器上。有显卡时都快一到两个数量级。

## 四、安装

### （一）拿到代码和权重

代码从 GitHub 仓库下载：

```bash
git clone https://github.com/Tarpelite/omniarch_pt.git
cd omniarch_pt
```

权重从北航网盘下载：

| 权重 | 大小 | 下载链接 |
|---|---|---|
| omniarch-base.pt | 1.2GB | LINK_BASE |
| omniarch-mini.pt | 293MB | LINK_MINI |
| omniarch-tiny.pt | 54MB | LINK_TINY |

下载后放进代码目录下的 `weights/` 文件夹。放了几个就按 base、mini、tiny 的顺序优先使用。不能访问 GitHub 的机器，直接拷贝整个代码文件夹和权重文件即可。

### （二）建 Python 环境

Linux 和 macOS：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows（PowerShell）：

```powershell
python -m venv .venv
.venv\Scripts\activate
```

国内网络建议先设清华镜像：

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### （三）装 PyTorch

按机器选一条：

| 机器 | 命令 |
|---|---|
| 没有显卡 | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| NVIDIA 显卡 | `pip install torch --index-url https://download.pytorch.org/whl/cu128` |
| Apple 芯片的 Mac | `pip install torch` |

RTX 50 系列显卡需要 PyTorch 2.7 以上。

### （四）装 OmniArch

```bash
pip install -e .
```

### （五）自检

```bash
omniarch info
```

看到参数量、设备（cpu、cuda 或 mps）和最小网格要求，说明装好了。

## 五、四种用法

### （一）网页演示

```bash
omniarch serve
```

浏览器打开 http://127.0.0.1:7860 。左侧选算例，调"给模型看的前几帧"和"往后预测的步数"，点"运行预测"。页面显示真值、预测和误差三栏，可以拖动时间轴，下方是误差随步数的曲线。左下角可以上传自己的数据。

要让同一局域网的其他电脑访问，改成 `omniarch serve --host 0.0.0.0 --port 7860`，再在防火墙放行 7860 端口。网页没有登录功能，只在内网开放。

### （二）命令行

```bash
omniarch examples                                   # 列出自带算例
omniarch predict 2d_turbulence                      # 预测，打印误差
omniarch predict 2d_turbulence --gif turb.gif       # 同时生成预测和真值对比动画
omniarch predict my_case.npz --steps 20             # 预测自己的数据
```

数据里帧数多于给定帧数时，`predict` 会把多出来的帧当真值，打印预测误差和对照误差。结果存成 `算例名_pred.npz`。

### （三）HTTP 接口

`omniarch serve` 启动后，其他程序可以直接调用。上传一个 npz，返回预测结果：

```bash
curl -F file=@my_case.npz "http://服务器IP:7860/api/forecast?steps=20" -o pred.npz
```

- 上传文件里的 `data` 是已有的最近若干帧；
- `steps` 是往后预测的步数；
- 返回的 `pred.npz` 里，`prediction` 是预测结果，形状（预测步数，物理量数，网格），`seconds` 是推演用时。

Python 里调用（调用方需要 `pip install requests numpy`）：

```python
import io, numpy as np, requests
buf = io.BytesIO(); np.savez(buf, data=frames.astype("float32"))
r = requests.post("http://服务器IP:7860/api/forecast", params={"steps": 20},
                  files={"file": ("case.npz", buf.getvalue())})
prediction = np.load(io.BytesIO(r.content))["prediction"]
```

### （四）Python 接口

和服务部署在同一台机器上时，可以直接调用模型：

```python
from omniarch import OmniArchPredictor
p = OmniArchPredictor("weights/omniarch-base.pt")      # 自动选显卡或 CPU
prediction = p.predict(frames, steps=20)               # frames: (帧数, 物理量数, 网格)
```

## 六、把自己的问题交给模型

### （一）整理数据

按下面四步把自己的仿真数据整理成输入：

**1. 物理量按固定顺序排列。**模型按位置区分物理量，不看名字。顺序要和训练时同类问题一致：

| 问题类型 | 物理量顺序 |
|---|---|
| 可压缩流 | 密度、Vx、（Vy、Vz）、压力 |
| 浅水 | 水深 |
| 二维扩散反应 | u、v |
| 单变量问题（平流、Burgers、扩散、反应扩散） | u |

**2. 放到等间距网格上。**网格尺寸尽量接近训练时的分辨率：一维 256 点，二维 64×64，三维 32³。原始网格更密时先降采样；最小不能低于一维 48 点、二维 32×32、三维 16³。周期边界的问题效果最好。

**3. 选好帧间隔。**相邻两帧之间，场的变化不能太小也不能太大。训练数据里，相邻两帧的平均相对变化大多在 1% 到 50% 之间。可以这样算：

```python
import numpy as np
d = np.abs(np.diff(frames, axis=0)).mean() / np.abs(frames).mean()
print(f"相邻帧平均相对变化 {d:.1%}")     # 远小于 1% 就隔帧取，远大于 50% 就加密输出
```

**4. 存成 npz。**`data` 形状为（帧数，物理量数，网格），float32；`channels` 选填，只用于网页显示：

```python
np.savez("my_case.npz", data=frames.astype("float32"), channels=np.array(["density", "Vx", "pressure"]))
```

PDEBench 格式的 HDF5 文件可以直接转换：`omniarch from-pdebench 文件.hdf5 --index 0 --out case.npz`，需要先 `pip install h5py`。

### （二）判断结果能不能信

模型总会给出一个结果，所以用之前要先验证。按代价从低到高：

1. **先拿一段已知答案的数据试。**用求解器多算 5 到 10 帧作为真值，放进 npz 一起上传。命令行和网页都会给出预测误差和对照误差。预测误差明显低于对照，才说明模型在这类问题上有效。
2. **看误差随步数的曲线。**平稳缓慢上升是正常的；突然跳升，说明从那一步起已经不可信。
3. **做物理检查。**看总质量、总能量是否明显漂移，有没有负密度、负压力、非物理的振荡。

### （三）效果不好时怎么办

- **缩短预测步数。**只用误差还小的前几步。
- **预测加校正。**每推若干步，用求解器从当前状态算一帧替换进去，再接着推。误差不会一直累积，求解器也只需要算其中一部分帧。这种用法我们还没有系统测过，效果要按上一小节的方法验证。
- **把数据反馈给开发方。**问题类型和训练数据差别大时，需要开发方用这类数据重新训练模型。请提供几十条完整的仿真轨迹，按上面的格式整理好。

## 七、作为服务长期运行

### （一）Linux systemd

新建 `/etc/systemd/system/omniarch.service`：

```ini
[Unit]
Description=OmniArch service
After=network.target

[Service]
User=omniarch
WorkingDirectory=/opt/omniarch_pt
ExecStart=/opt/omniarch_pt/.venv/bin/omniarch serve --host 0.0.0.0 --port 7860
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

然后 `sudo systemctl enable --now omniarch`。

### （二）Docker

```bash
docker build -t omniarch-pt --build-arg PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple .
docker run -d -p 7860:7860 -v $(pwd)/weights:/app/weights omniarch-pt
```

镜像默认装 CPU 版 PyTorch。用显卡时构建加 `--build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128`，运行加 `--gpus all`。

### （三）放到域名下

前面加 nginx 反向代理，把域名转到 `127.0.0.1:7860`。需要访问控制时在 nginx 上加密码或只放内网。服务同一时间只算一个请求，其余请求排队。

## 八、常见问题

| 现象 | 处理 |
|---|---|
| `omniarch info` 显示 cpu，但机器有显卡 | 装成了 CPU 版 PyTorch，按第四节重装 CUDA 版；`nvidia-smi` 确认驱动正常 |
| 报 `no checkpoint found` | 权重没放进 `weights/`，或用 `--ckpt 路径` 指定，或设环境变量 `OMNIARCH_CKPT` |
| 报 `grid size ... too small` | 网格小于最小要求，插值加密后再预测 |
| 报 `at most 8 fields` | 物理量超过 8 个，拆成几组分别预测 |
| 预测明显不对 | 先查物理量顺序和帧间隔，见第六节 |
| 显存不够 | 三维大网格时出现，改用 `--device cpu`，或先降采样 |
| 端口被占用 | `omniarch serve --port 7861` |
| 离线机器安装 | 在联网机器上 `pip download -d pkgs torch numpy matplotlib fastapi uvicorn python-multipart`，拷过去后 `pip install --no-index --find-links pkgs ...` |

## 九、命令速查

| 命令 | 作用 |
|---|---|
| `omniarch info` | 查看模型和设备 |
| `omniarch examples` | 列出自带算例 |
| `omniarch predict 算例或文件 [--history K] [--steps N] [--gif 文件]` | 预测并评测 |
| `omniarch from-pdebench 文件 --out case.npz` | 从 PDEBench 原始文件截一条轨迹 |
| `omniarch serve [--host] [--port]` | 启动网页演示和 HTTP 接口 |
| 任一命令加 `--device cpu` 或 `--ckpt 路径` | 指定设备或权重 |
