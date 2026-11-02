# 如何复跑本次验收

前置:任意一台 dev A100(4 卡)机器,能访问 `/nfs-data`。

```bash
# 1) 准备(代码 + pytest/ruff + 运行器;只需一次)
#    /data/lx2v-sync 是待验收代码副本,/data/verify-logs 放结果
scp -r . dev-gpustack-a100-000X:/data/lx2v-sync
#    pytest/ruff 从 mirrors.aliyun.com 装进挂载目录(镜像内没有,机器也不能直连 PyPI)
docker run --rm -v /data/lx2v-pydeps:/opt/pydeps <IMG> \
  bash -lc 'pip install --no-cache-dir -q --target /opt/pydeps -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com pytest ruff==0.11.0'

# 2) 静态门禁
python docs/上游同步-2026-11/audit_our_config_keys.py            # 必须 rc=0
docs/上游同步-2026-11/evidence/ws8-gate-config-keys.sh           # 必须 rc=0
git diff --name-only upstream/main HEAD -- lightx2v lightx2v_platform | grep -v -E '<白名单>'
python3 -m pytest test_cases/ -q                                  # 92 passed
ruff==0.11.0 check / format --check lightx2v lightx2v_platform test_cases deploy

# 3) 起服务(new = 我们的代码,old = 现网镜像自带代码)
docs/上游同步-2026-11/verify/serve.sh new seedvr2-3b/seg121-4card 4 18000 lx2v_new
docs/上游同步-2026-11/verify/serve.sh old seedvr2-3b/seg121-4card 4 18001 lx2v_old

# 4) 发请求(第 5 个参数 = 起跑 N 秒后 DELETE,做取消验证)
docs/上游同步-2026-11/verify/serve.sh  # 见上
docs/上游同步-2026-11/verify/sr.sh 18000 \
  /nfs-data/lx2v_test/assets/sr_768p_15s.mp4 /nfs-data/lx2v_test/verify-out/x.mp4 '"sr_ratio":4.0'

# 5) 量画质/尺寸
python3 docs/上游同步-2026-11/verify/measure.py info  out.mp4
python3 docs/上游同步-2026-11/verify/measure.py psnr  old.mp4 new.mp4
python3 docs/上游同步-2026-11/verify/measure.py sharp out.mp4
python3 docs/上游同步-2026-11/verify/measure.py judder out.mp4
```

注意:
- `serve.sh` 的 `new` 模式把代码挂到 `/opt/LightX2V`、decord 垫片挂到 `/opt/decord_shim`,
  与 `old`(镜像自带)构成「同镜像换代码」对照。
- 4 卡 SeedVR 每个 rank 在 host 侧要 ~18-22G RSS;dev 集群的机器普遍已被邻居占掉 83~107G,
  起服务前先看 `free -g` 的 available,必要时加 `-e MALLOC_ARENA_MAX=2 -e OMP_NUM_THREADS=8`。
