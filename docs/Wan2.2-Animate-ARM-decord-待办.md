# Wan2.2-Animate 接入待办(ARM/鲲鹏920)

> 状态:**暂缓**。Animate 是这批模型里最重的一个,投入产出比低,等 I2V / S2V / 文生图都测完、确实要上 animate 时再单独搞。
> 记录时间:2026-06-28

## 一、Animate 需要什么

1. **输入两样**
   - `video_path` = 驱动视频(一个人在动 → 提供姿态/表情/动作)
   - `refer_path` = 角色参考图(要动画化/被替换的人,如 `girl.png`)
2. **预处理一步**(`tools/preprocess/preprocess_data.py`)
   - 从 视频+图 提取 `src_pose.mp4` / `src_face.mp4` / `src_ref.png`,再喂给推理
   - 需要 `process_checkpoint`(预处理模型,在 `$model_path/process_checkpoint`),要确认下载的 `Wan2.2-Animate-14B` 里有没有,没有要补下
3. **驱动视频来源**
   - 官方仓库 [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) / [ModelScope 页](https://www.modelscope.cn/models/Wan-AI/Wan2.2-Animate-14B) 的 examples/studio 里通常带示例
   - 或任意一段人在跳舞/做动作的短视频(手机录、stock 片段)

## 二、硬阻塞:decord 在 ARM 上装不了

### 根因(不是漏装,是 pip 路走不通)

`dockerfiles/Dockerfile_aarch64_cu128:54` 的 pip 列表里其实**写了 `decord`**,但 PyPI 上:

| 包 | Linux aarch64 wheel? |
|---|---|
| `decord` 0.6.0 | ❌ 只有 `manylinux2010_x86_64`(纯 x86) |
| `eva-decord` 0.6.1(社区 fork) | ❌ Linux 只有 x86;arm64 只有 macOS |

本机是**鲲鹏920 aarch64**,`pip install decord` 找不到匹配 wheel → 整条 RUN 报错或包被跳过 → 镜像里没有。**pip 这条路在 ARM 上无解,只能源码编译。**

### 正确做法:源码编译(改 `Dockerfile_aarch64_cu128`)

**1. 补 ffmpeg 开发库**(当前只装了 `ffmpeg` 运行时,缺 `-dev` 头文件,编译会失败)。加到第 37 行那批 apt 里:

```dockerfile
libavcodec-dev libavfilter-dev libavformat-dev libavutil-dev \
libswresample-dev libswscale-dev libavdevice-dev
```

**2. 把 pip 列表(第 54 行)里的 `decord` 删掉**,换成源码编译块:

```dockerfile
# decord (ARM 无 wheel,源码编译;USE_CUDA=0 用 CPU 解码即可)
RUN git clone --recursive https://github.com/dmlc/decord /opt/decord && \
    cd /opt/decord && mkdir build && cd build && \
    cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    cd ../python && pip install --no-cache-dir -e . && \
    python -c "import decord; print('decord OK', decord.__version__)"
```

`-e .` 装在 `/opt/decord/python`,**该目录别删**。

**3. 验证**(容器内):

```bash
python -c "from decord import VideoReader; print('ok')"
```

## 三、decord 的影响范围(哪些模型受阻)

代码里 hard-import decord 的路径:
- `wan_animate_runner.py`(animate)
- `wan_s2v_runner.py` 的 `src_pose_path` 输入
- `wan_infinitetalk_runner.py` 的视频 cond 输入
- `vace_processor.py`(vace)
- `tools/preprocess/*`、`worldplay_ar_dataset.py`

**不受影响、可先测**:I2V、S2V(图+音频,不走 src_pose)、文生图(Qwen-Image / Z-Image)。

## 四、Animate 上线时的完整清单

- [ ] 重出带 decord 的 ARM 镜像(上面第二节)
- [ ] 确认/补下 `process_checkpoint` 预处理模型
- [ ] 准备驱动视频(官方示例或自录)
- [ ] 跑预处理 `tools/preprocess/preprocess_data.py` → src_pose/src_face/src_ref
- [ ] 跑推理(注意 `replace_flag` 切 animation/换人两模式;720×1280/77帧)
