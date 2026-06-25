# sub2api vs new-api 深度对比分析

> 分析日期：2026-03-06
> 分析版本：sub2api（本地源码）vs new-api（本地源码）

---

## 一、项目定位

| 维度 | sub2api | new-api |
|------|---------|---------|
| **核心定位** | 企业级 AI API 网关平台（MaaS） | 通用 LLM 网关与 AI 资产管理系统 |
| **目标用户** | 需要账号池管理 + 精细计费的 SaaS 运营者 | 需要统一 AI 入口的个人/企业/开发者 |
| **设计哲学** | 深度整合单一/少数账号体系，精细化调度 | 广泛接入 40+ 供应商，横向扩展 |
| **开源许可** | 未明确（私有项目） | AGPLv3 |
| **前端框架** | Vue 3 + Vite + Pinia | React 18 + Vite + Semi Design |

---

## 二、代码量统计

### 2.1 后端 Go 代码

| 指标 | sub2api | new-api | 倍数 |
|------|---------|---------|------|
| **业务代码行数** | 275,730 行 | 88,776 行 | **3.1x** |
| **业务代码文件数** | 596 个 | 492 个 | 1.2x |
| **测试代码行数** | 103,319 行 | 5,208 行 | **19.8x** |
| **测试文件数** | 325 个 | 18 个 | **18x** |
| **Go 依赖数量** | 178 个 | 129 个 | 1.4x |

> sub2api 的 275K 行业务代码中包含约 **131,755 行 Ent ORM 生成代码**，手写业务代码约 **144K 行**，仍是 new-api 的 **1.6 倍**。

### 2.2 按层分布

```
sub2api（业务逻辑层最重）          new-api（中继层最重）
────────────────────────           ──────────────────────
handler     25,298 行              relay       30,708 行  ← 核心
service     78,331 行  ← 核心      controller  17,875 行
repository  22,753 行              service      9,496 行
pkg          8,536 行              model        9,947 行
Ent 生成   131,755 行              middleware   2,077 行
```

sub2api 的 service 层（78K）单独比 new-api 整个后端（89K）还大，说明其核心业务逻辑（账号调度/计费/并发）复杂度远超 new-api。

### 2.3 前端代码

| 指标 | sub2api | new-api |
|------|---------|---------|
| **行数** | 92,809 行 | 96,191 行 |
| **文件数** | 275 个 | 372 个 |
| **框架** | Vue 3 | React 18 |

前端规模基本相当，new-api 文件更多（页面功能更多）。

---

## 三、技术栈对比

### 3.1 后端核心技术栈

| 维度 | sub2api | new-api |
|------|---------|---------|
| **语言版本** | Go 1.25.7 | Go 1.25.1 |
| **Web 框架** | Gin 1.9.1 | Gin 1.9.1 |
| **ORM** | **Ent 0.14.5**（代码生成） | **GORM 1.25**（运行时反射） |
| **数据库支持** | PostgreSQL（唯一） | SQLite / MySQL / PostgreSQL |
| **依赖注入** | **Wire**（编译时） | 无 DI 框架 |
| **日志** | **Zap**（结构化） | 自定义日志模块 |
| **缓存** | Redis v9 + Ristretto（二级缓存） | Redis v8 + samber/hot（LRU） |
| **并发池** | pond/v2 | gopool |
| **JSON** | 标准库 | goccy/go-json（高性能）+ gjson/sjson |
| **TLS** | **utls**（自定义指纹） | 标准 crypto/tls |
| **HTTP/2** | **h2c 支持** | 无 |
| **Token 计数** | 无 | **tiktoken-go**（精确计数） |
| **测试** | **testcontainers**（真实 DB 容器） | 无容器测试 |
| **安全扫描** | govulncheck + gosec | 无 |

### 3.2 前端核心技术栈

| 维度 | sub2api | new-api |
|------|---------|---------|
| **框架** | Vue 3.4 | React 18 |
| **状态管理** | Pinia | 无（useState/Context） |
| **UI 组件库** | TailwindCSS（原子化） | Semi Design（字节跳动） |
| **包管理** | pnpm（强制） | Bun（推荐） |
| **国际化** | vue-i18n（中/英 2 语言） | i18next（7 种语言） |
| **TypeScript** | 全量 | 全量 |

### 3.3 独有技术选型

**sub2api 独有：**
- `utls`：TLS 客户端指纹伪造（绕过 AI 平台检测）
- `Wire`：编译时依赖注入（Google 出品）
- `testcontainers`：集成测试用真实 PostgreSQL/Redis 容器
- `Ent`：Meta 出品的类型安全 ORM，schema 即代码
- h2c（HTTP/2 Cleartext）支持
- `pond/v2`：高性能协程池（精细并发控制）

**new-api 独有：**
- `tiktoken-go`：OpenAI tiktoken 算法精确计数 token
- `stripe-go`：Stripe 支付 SDK
- `go-webauthn`：Passkey/WebAuthn 认证
- `pyroscope-go`：持续性能剖析（Grafana 生态）
- `gjson/sjson`：高性能 JSON 读写（无需反序列化）
- `goccy/go-json`：高性能 JSON 序列化（替代标准库）
- 多媒体处理：音频（WAV/MP3/FLAC）+ 视频（MP4）解析库

---

## 四、支持的 AI 平台/模型

| 平台 | sub2api | new-api |
|------|---------|---------|
| **Anthropic Claude** | 深度支持（OAuth/API Key/Setup Token） | 支持 |
| **OpenAI GPT** | 支持 | 支持 |
| **Google Gemini** | 深度支持（含 CLI Session/Drive） | 支持 |
| **Antigravity** | 独特支持（专有平台） | 不支持 |
| **Sora** | 支持（媒体签名 URL） | 不支持 |
| **AWS Bedrock** | 不支持 | 支持 |
| **国内大模型** | 不支持 | 百度/阿里/智谱/讯飞/腾讯/360 等全线 |
| **DeepSeek** | 不支持 | 支持 |
| **xAI (Grok)** | 不支持 | 支持 |
| **Midjourney** | 不支持 | 支持（通过 Proxy） |
| **Suno（音乐生成）** | 不支持 | 支持 |
| **自建模型（MaaS）** | Custom OpenAI（GPUStack/vLLM） | Ollama/Xinference |
| **支持供应商总数** | **6 类** | **58 类（40+ 供应商）** |

---

## 五、账号管理模式（核心差异）

这是两个项目**最本质的区别**。

### sub2api：账号池 + 智能调度

```
核心概念：Account（AI 平台账号）是一等公民
- 一个账号对应 OAuth/API Key/SetupToken/Upstream 等凭据
- 多账号组成"分组"，智能调度请求

调度算法：SelectAccountWithLoadAwareness
  1. 过滤：排除禁用/过期/速率限制中的账号
  2. 计算负载 = (并发数 - 可用并发) / 优先级
  3. 选择最低负载账号
  4. Sticky Session：同用户倾向复用同账号（提高对话一致性）

错误转移：
  - 单账号失败 → 自动切换，最多10次
  - 特定错误可透传（ErrorPassthroughService）
```

### new-api：渠道（Channel）+ 权重分发

```
核心概念：Channel（API 渠道）是一等公民
- 渠道按 Type 区分（58种），每渠道可配多个 Key
- 渠道按 Weight/Priority 分配请求

分发策略：
  1. 权重随机（Weighted Random）
  2. 优先级排序
  3. 分组隔离（Group）
  4. Multi-Key 模式：一个渠道配多个 Key，轮询

特有功能：
  - 渠道亲和度（用户偏好某渠道）
  - 跨分组重试（CrossGroupRetry）
  - 自动禁用（AutoBan）
```

**一句话：**
- sub2api 侧重**同平台多账号**的精细调度
- new-api 侧重**多平台多渠道**的广覆盖路由

---

## 六、计费系统对比

| 计费维度 | sub2api | new-api |
|---------|---------|---------|
| **计费粒度** | Token 级实时计费（流式响应中即时计算） | Token 级（请求完成后） |
| **定价层级** | 3级：分组自定义 → 动态价格 → 兜底价格 | 渠道级倍率 × 模型价格 |
| **用户余额** | 美元余额 | 额度（Quota）= $0.002/1K tokens |
| **订阅系统** | 订阅额度（独立于余额） | 订阅计划（周期重置） |
| **API Key 子额度** | Key 级别 Quota | Token 级别 RemainQuota |
| **缓存计费** | 不支持 | 支持（OpenAI/Azure/DeepSeek/Claude/Qwen） |
| **支付集成** | SMTP 邮件（仅重置密码） | Stripe + Epay + Creem |
| **邀请奖励** | 不支持 | AffCode 邀请返利系统 |
| **充值系统** | 不支持 | TopUp 订单 + 兑换码 |

**sub2api 计费特色：**
- 流式响应中实时提交 Usage，不等待完整响应
- WorkerPool 异步计费（不阻塞主请求路径）
- `simple` 模式可完全跳过计费（内部使用）
- 动态价格数据：从 GitHub 定期拉取，无需重启更新

**new-api 计费特色：**
- 完整的 SaaS 商业化套件（充值/订阅/邀请）
- 配额显示支持 USD/CNY/Token 三种单位
- 缓存计费（prefix cache 折扣）

---

## 七、用户与认证体系

| 认证维度 | sub2api | new-api |
|---------|---------|---------|
| **Web 登录** | 用户名/密码 + TOTP 2FA | 用户名/密码 + 2FA + **Passkey (WebAuthn)** |
| **OAuth 登录** | 有限 | GitHub/Discord/OIDC/LinuxDO/微信/Telegram/自定义 |
| **用户角色** | 管理员/普通用户 | Root/Admin/Common/Guest 4 级 |
| **分组管理** | 账号分组决定可调度的账号池 | 渠道分组决定可用渠道 |
| **IP 白名单** | API Key 级别 | Token 级别 |
| **并发控制** | 用户级 + 账号级双重并发限制 | 无精细并发控制 |
| **速率限制** | RPM/TPM 多时间窗口（小时/天/周） | 全局速率限制 |

---

## 八、部署与运维

| 维度 | sub2api | new-api |
|------|---------|---------|
| **最低配置** | PostgreSQL + Redis（必须） | SQLite（单文件，零依赖） |
| **Docker** | Docker Compose | Docker / Docker Compose |
| **一键安装** | Linux Shell 脚本 + Systemd | Docker 命令 / BaoTa 面板 |
| **初始化向导** | Setup Wizard（首次启动引导） | 需手动配置 |
| **配置方式** | YAML 文件 + 环境变量 | 纯环境变量 |
| **前端嵌入** | 编译时嵌入（`-tags embed`） | 编译时嵌入 |
| **性能监控** | 无 | Pyroscope + Google Analytics + Umami |
| **国际化** | 中/英（2 语言） | 中/繁中/英/法/日/俄/越（7 语言） |

---

## 九、代码质量对比

### 9.1 测试覆盖（最直接指标）

| 项目 | 测试文件数 | 测试代码行数 | 测试类型 |
|------|-----------|------------|---------|
| **sub2api** | 325 个 | 103,319 行 | 单元 + 集成 + Benchmark |
| **new-api** | 18 个 | 5,208 行 | 仅单元测试 |

sub2api 测试文件数是 new-api 的 **18 倍**，测试代码量是 **20 倍**。

### 9.2 静态分析

**sub2api** 有详细的 `.golangci.yml`（600+ 行），启用 7 个 linter：

```yaml
linters:
  - depguard    # 依赖层级隔离（违反即 CI 失败）
  - errcheck    # 错误不可忽略（含类型断言）
  - gosec       # 安全扫描
  - govet       # Go 官方检查
  - ineffassign # 无效赋值检测
  - staticcheck # 100+ 条静态分析规则（SA/ST/S/QF 全系列）
  - unused      # 死代码检测
```

还有架构边界的**硬性约束**（Lint 强制执行，不通过则 CI 失败）：
```yaml
# service 层不能 import repository / gorm / redis
# handler 层不能 import repository / gorm / redis
```

**new-api** 无 `.golangci.yml`，没有任何静态分析配置。

### 9.3 架构纪律

| 维度 | sub2api | new-api |
|------|---------|---------|
| **依赖方向** | Lint 强制（违反 CI 失败） | 无约束，controller 直接用 model |
| **依赖注入** | Wire 编译时 DI（依赖关系显式声明） | 无 DI，全局变量/函数调用 |
| **ORM 类型安全** | Ent（代码生成，编译期类型检查） | GORM（运行时反射，错误只在运行时暴露） |
| **数据库迁移** | 顺序编号 SQL 文件（可审计、可回滚） | GORM AutoMigrate（自动魔法迁移） |
| **软删除** | SoftDeleteMixin 全局统一 | 无统一机制 |

### 9.4 CI/CD 流水线

**sub2api** 有三条独立 CI 工作流：
- `backend-ci.yml`：单元测试 + 集成测试 + golangci-lint
- `security-scan.yml`：govulncheck + gosec + pnpm audit
- `release.yml`：tag 触发多平台构建

**new-api** CI 配置较基础。

---

## 十、特色功能对比

### sub2api 独有特色

1. **Antigravity 平台支持**：专有 AI 平台接入，支持混合调度（Claude + Antigravity 共用端点）
2. **Sora 媒体签名 URL**：生成临时签名媒体链接（防盗链），TTL 控制（默认 900 秒）
3. **Gemini CLI Session**：Google Drive 文件访问，长上下文会话支持
4. **并发等待队列**：超并发时自动排队，用户消息串行化（UserMessageQueueService）
5. **HTTP/2 (h2c) 支持**：Cleartext HTTP/2，并发流控制（max_concurrent_streams）
6. **Simple Mode（简易模式）**：跳过所有计费检查，适合内部/开发使用
7. **TLS 指纹自定义（utls）**：绕过 TLS 指纹检测
8. **动态价格数据**：从 GitHub 定期拉取最新价格表，无需重启即可更新定价

### new-api 独有特色

1. **完整商业化套件**：Stripe/Epay/Creem 支付 + 兑换码 + 邀请返利 + 订阅计划管理
2. **Passkey (WebAuthn) 支持**：无密码登录，生物识别认证
3. **缓存计费**：Prefix Cache 折扣支持，多供应商缓存计费规则
4. **格式自动转换**：OpenAI ⇄ Claude Messages、OpenAI → Gemini、Thinking 内容格式转换
5. **多媒体处理**：音频格式（WAV/MP3/FLAC/OGG）+ 视频格式处理
6. **Rerank 接口**：Cohere/Jina Rerank 支持
7. **国内大模型生态**：百度/阿里/智谱/讯飞等全线接入
8. **Midjourney/Suno 集成**：图像/音乐生成服务代理

---

## 十一、适用场景总结

**选择 sub2api 如果你需要：**
- 管理多个 **Claude/Gemini/OpenAI 官方账号**，精细调度避免封号
- 构建**企业级 AI API 代理服务**，需要严格的计费和并发控制
- 接入 **Antigravity** 等特殊平台
- 需要高可靠的**账号级容错转移**
- 项目架构要求**代码质量和可维护性**较高

**选择 new-api 如果你需要：**
- 快速搭建支持 **40+ AI 供应商**的统一网关
- 需要完整的**商业化 SaaS 能力**（支付/订阅/邀请）
- 接入**国内大模型**（百度/阿里/智谱等）
- 需要 **SQLite 单文件**轻量部署
- 需要 **Midjourney/Suno** 等多模态服务代理
- 国际化要求高（7 语言）

---

## 十二、总结

| 维度 | sub2api | new-api |
|------|---------|---------|
| **手写业务代码** | ~144K 行 | ~89K 行 |
| **测试代码** | 103K 行（325 文件） | 5K 行（18 文件） |
| **供应商支持** | 6 类（深度） | 58 类（广度） |
| **工程严谨性** | 高（Lint 强制分层、编译期类型安全、容器集成测试） | 一般（无 Lint 配置、无集成测试） |
| **商业化能力** | 弱（无支付） | 强（Stripe/Epay/Creem + 完整 SaaS 套件） |
| **部署门槛** | 高（必须 PostgreSQL + Redis） | 低（SQLite 单文件可用） |
| **账号管理** | 账号池智能调度（精细） | 渠道权重分发（广覆盖） |
| **并发控制** | 用户级 + 账号级双重限制 | 无精细并发控制 |
| **代码质量** | 显著更高 | 一般 |

**本质区别：**
- sub2api 是**深度专注**的账号调度平台，工程质量优先，功能聚焦
- new-api 是**广度优先**的 AI 网关聚合器，功能覆盖优先，快速迭代
