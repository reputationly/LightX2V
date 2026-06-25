# GPUStack + new-api 集成方案分析

> 文档版本：v1.0 | 更新日期：2026-03-16

---

## 目录

1. [背景与目标](#1-背景与目标)
2. [架构对比](#2-架构对比)
3. [功能层分析](#3-功能层分析)
4. [代码层分析](#4-代码层分析)
5. [集成架构设计](#5-集成架构设计)
6. [部署方案](#6-部署方案)
7. [Channel 配置指南](#7-channel-配置指南)
8. [用户与配额管理](#8-用户与配额管理)
9. [高级功能](#9-高级功能)
10. [迁移路径](#10-迁移路径)
11. [验证方案](#11-验证方案)

---

## 1. 背景与目标

### 1.1 为何迁移

**sub2api** 是一个轻量级 OpenAI API 代理，功能相对单一：

- 仅做请求转发，无用户管理
- 不支持多 Channel 负载均衡
- 无计费与配额系统
- 无 Web 管理界面
- Token 管理能力弱

随着 GPUStack 集群规模扩大，团队需要：

| 需求 | sub2api | new-api |
|------|---------|---------|
| 多节点负载均衡 | 不支持 | 支持（Weight/Priority） |
| 用户 Token 隔离 | 不支持 | 支持（多租户） |
| 计费与用量统计 | 不支持 | 支持（按 Token 计费） |
| Web 管理后台 | 不支持 | 支持 |
| 模型名称映射 | 不支持 | 支持（Model Mapping） |
| 多后端类型 | 不支持 | 支持（58 种 Channel 类型） |
| 速率限制 | 不支持 | 支持（Group/Token 级） |

**迁移目标**：以 new-api 作为统一 AI 网关，GPUStack 集群作为后端 Channel，实现企业级多租户 LLM 服务管理。

### 1.2 技术路线

```
用户 / 应用
    │
    │  OpenAI-compatible API
    ▼
┌─────────────┐
│   new-api   │  ← 统一网关（认证、计费、路由、限速）
│  :3000      │
└─────┬───────┘
      │  OpenAI-compatible API (转发)
      ▼
┌─────────────────────────────┐
│        GPUStack Cluster     │
│  :80  (OpenAI-compatible)   │
│  ┌──────┐ ┌──────┐ ┌──────┐│
│  │Worker│ │Worker│ │Worker││
│  │ GPU0 │ │ GPU1 │ │ GPU2 ││
│  └──────┘ └──────┘ └──────┘│
└─────────────────────────────┘
```

---

## 2. 架构对比

### 2.1 sub2api 架构（现状）

```
Client → sub2api → GPUStack
          │
          └── 简单代理，无状态
              无用户隔离
              无计费
```

**局限性**：
- 所有用户共享同一个 API key
- 无法限制某个用户的用量
- 无法区分不同模型的费用
- 单 GPUStack 节点，无冗余

### 2.2 new-api 架构（目标）

```
Client A (Token A) ──┐
Client B (Token B) ──┤──→ new-api ──→ Channel 1 (GPUStack Node 1, Weight=3)
Client C (Token C) ──┘         │
                                └──→ Channel 2 (GPUStack Node 2, Weight=1)
                                │         (Priority Fallback)
                                └──→ Channel 3 (GPUStack Node 3, Backup)
```

**new-api 职责分层**：

| 层次 | 功能 |
|------|------|
| 认证层 | Token 验证、IP 白名单 |
| 路由层 | Model → Channel 映射、负载均衡 |
| 计费层 | Token 计数、费率换算、余额扣减 |
| 适配层 | 请求格式转换（Anthropic/Gemini/etc → OpenAI） |
| 缓存层 | 语义缓存（可选） |
| 日志层 | 请求日志、用量统计 |

---

## 3. 功能层分析

### 3.1 Channel 类型系统（58 种）

new-api 支持 58 种 Channel 类型，每种对应不同的 API 格式适配器。对于 GPUStack 集成，使用：

- **Type 1**：`OpenAI`（最优选择，GPUStack 完整兼容 OpenAI API 格式）

其他常见类型供参考：

| Type | 名称 | 适用场景 |
|------|------|----------|
| 1 | OpenAI | GPUStack、vLLM、LocalAI 等兼容 OpenAI 的后端 |
| 3 | Azure OpenAI | Azure 托管模型 |
| 8 | 自定义 | 特殊格式后端 |
| 14 | Anthropic | Claude API |
| 24 | Gemini | Google Gemini |
| 40 | Ollama | 本地 Ollama |

### 3.2 核心功能模块

#### 请求处理流水线

```
请求入口 /v1/chat/completions
    ↓
TokenAuth 中间件（验证 Bearer Token）
    ↓
Distributor 中间件（选择 Channel）
    ↓
Relay Handler（格式转换 + 转发）
    ↓
计费记录（LogUsage）
    ↓
返回响应
```

#### 计费系统

- 按 Prompt/Completion Token 分别计费
- 支持自定义倍率（ModelRatio × CompletionRatio）
- 支持 Group 折扣率
- 实时余额扣减（预扣 + 结算）

#### 用户管理

- **Token**：API key，可设置过期时间、使用次数上限、模型白名单
- **Group**：用户分组，设置不同费率和权限
- **Channel**：后端 AI 服务节点，含健康检测

---

## 4. 代码层分析

### 4.1 项目结构

```
new-api/
├── main.go                    # 入口，初始化 DB、Router、定时任务
├── router/
│   ├── relay-router.go        # /v1/* 路由注册
│   └── api-router.go          # 管理 API 路由
├── middleware/
│   ├── auth.go                # Token 认证中间件
│   └── distributor.go         # Channel 选择器（核心）
├── relay/
│   ├── relay_adaptor.go       # 适配器工厂（按 Channel Type 分发）
│   ├── controller/
│   │   └── relay.go           # Relay 主控制器
│   └── channel/
│       ├── openai/            # OpenAI 格式适配器（GPUStack 使用此路径）
│       │   ├── adaptor.go
│       │   ├── main.go
│       │   └── model.go
│       ├── anthropic/         # Anthropic 格式适配器
│       └── ...                # 其他 58 种适配器
├── model/
│   ├── channel.go             # Channel 数据模型
│   ├── token.go               # Token 数据模型
│   └── user.go                # User 数据模型
├── constant/
│   └── channel.go             # 58 种 Channel 类型常量定义
├── common/
│   └── config.go              # 全局配置
├── docker-compose.yml         # Docker 部署配置
└── .env.example               # 环境变量示例
```

### 4.2 关键代码路径

#### Channel 类型常量 (`constant/channel.go`)

```go
const (
    ChannelTypeOpenAI          = 1   // OpenAI (兼容 vLLM/GPUStack)
    ChannelTypeAzure           = 3
    ChannelTypeAnthropic       = 14
    ChannelTypeGemini          = 24
    ChannelTypeOllama          = 40
    // ... 共 58 种
)
```

GPUStack 使用 `ChannelTypeOpenAI = 1`，因为 GPUStack 暴露完整的 OpenAI-compatible REST API。

#### Channel 数据模型 (`model/channel.go`)

```go
type Channel struct {
    Id                 int       `json:"id"`
    Type               int       `json:"type"`           // Channel 类型，填 1
    Key                string    `json:"key"`            // GPUStack API Key (gsk_xxxx)
    BaseURL            string    `json:"base_url"`       // GPUStack 地址，如 http://gpustack:80
    Models             string    `json:"models"`         // 模型列表，逗号分隔
    ModelMapping       *string   `json:"model_mapping"`  // 模型名称映射 JSON
    Weight             *uint     `json:"weight"`         // 负载均衡权重
    Priority           *int64    `json:"priority"`       // 优先级（高优先）
    Status             int       `json:"status"`         // 1=启用, 2=禁用, 3=自动禁用
    ResponseTime       int       `json:"response_time"`  // 响应时间（健康检测用）
    TestTime           time.Time `json:"test_time"`
    UsedQuota          int64     `json:"used_quota"`
}
```

#### Distributor 中间件 (`middleware/distributor.go`)

```go
// 核心逻辑：根据请求的 model 选择合适的 Channel
func Distribute() gin.HandlerFunc {
    return func(c *gin.Context) {
        // 1. 从请求中提取 model 名称
        // 2. 根据 model 查找支持该 model 的所有 Channel
        // 3. 按 Priority 过滤（选最高优先级的组）
        // 4. 在同优先级内按 Weight 加权随机选择
        // 5. 将选中的 Channel 注入 Context
        c.Next()
    }
}
```

#### 适配器工厂 (`relay/relay_adaptor.go`)

```go
func GetAdaptor(apiType int) adaptor.Adaptor {
    switch apiType {
    case constant.APITypeOpenAI:
        return &openai.Adaptor{}    // GPUStack 走这条路径
    case constant.APITypeAnthropic:
        return &anthropic.Adaptor{}
    case constant.APITypeGemini:
        return &gemini.Adaptor{}
    // ...
    }
}
```

#### OpenAI 适配器 (`relay/channel/openai/adaptor.go`)

GPUStack 集成使用此适配器，关键方法：

```go
func (a *Adaptor) GetRequestURL(info *relaycommon.RelayInfo) (string, error) {
    // 拼接 BaseURL + 路径
    // 例：http://gpustack:80/v1/chat/completions
    return fmt.Sprintf("%s/v1/%s", info.BaseUrl, info.RequestURLPath), nil
}

func (a *Adaptor) SetupRequestHeader(c *gin.Context, req *http.Request, info *relaycommon.RelayInfo) error {
    // 注入 Authorization: Bearer <gpustack_api_key>
    req.Header.Set("Authorization", "Bearer "+info.ApiKey)
    return nil
}
```

### 4.3 请求转发全链路

以 `POST /v1/chat/completions` 为例：

```
1. relay-router.go        注册路由 → RelayHandler
2. middleware/auth.go     验证用户 Token（Bearer sk-xxx）
3. middleware/distributor.go  根据 model 选择 GPUStack Channel
4. relay/controller/relay.go  判断请求类型（chat/embedding/etc）
5. relay_adaptor.go       获取 OpenAI Adaptor（type=1）
6. openai/adaptor.go      构建转发请求：
                           URL = http://gpustack:80/v1/chat/completions
                           Header: Authorization: Bearer gsk_xxxx
                           Body: 原始请求体（直接透传）
7. 发送 HTTP 请求到 GPUStack
8. 流式/非流式响应处理
9. 计费：统计 prompt/completion tokens，扣减余额
10. 返回响应给客户端
```

---

## 5. 集成架构设计

### 5.1 完整架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        Client Layer                          │
│  App A    App B    App C    SDK    curl                      │
│  (Token A)(Token B)(Token C)(Token D)(Token E)               │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTPS :443 / HTTP :3000
                       │ POST /v1/chat/completions
                       │ Authorization: Bearer sk-xxx
┌──────────────────────▼──────────────────────────────────────┐
│                      new-api Gateway                         │
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │  Auth Layer │  │ Route Layer  │  │  Billing Layer    │  │
│  │  Token验证  │  │ Model→Channel│  │  Token计费+扣余额 │  │
│  │  IP白名单   │  │ Weight路由   │  │  用量日志         │  │
│  └─────────────┘  └──────────────┘  └───────────────────┘  │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              OpenAI Adaptor (Type=1)                  │  │
│  │  请求透传 + Header注入(Authorization: Bearer gsk_xxx) │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────┬─────────────────────┬─────────────────────┘
                  │                     │
     Priority=10  │          Priority=5 │
     Weight=3     │          Weight=1   │
                  ▼                     ▼
┌─────────────────────────┐  ┌─────────────────────────┐
│   GPUStack Node 1       │  │   GPUStack Node 2       │
│   http://gpu1:80        │  │   http://gpu2:80        │
│                         │  │                         │
│  Models:                │  │  Models:                │
│  - Qwen2.5-72B-Instruct │  │  - Llama-3.1-8B        │
│  - Qwen2.5-32B-Instruct │  │  - Qwen2.5-7B-Instruct │
│                         │  │                         │
│  Workers:               │  │  Workers:               │
│  ├── A100 x2            │  │  ├── RTX4090 x2         │
│  └── A100 x2            │  │  └── RTX4090 x1         │
└─────────────────────────┘  └─────────────────────────┘
```

### 5.2 数据流说明

| 步骤 | 组件 | 操作 |
|------|------|------|
| 1 | Client | 携带 `sk-user-token` 发起请求 |
| 2 | new-api Auth | 验证 Token 有效性、余额、模型权限 |
| 3 | new-api Distributor | 根据 `model` 字段选择 Channel（GPUStack 节点） |
| 4 | new-api OpenAI Adaptor | 替换 Authorization Header 为 GPUStack API Key |
| 5 | GPUStack | 接收请求，内部负载均衡到可用 Worker |
| 6 | GPUStack Worker | 推理引擎（vLLM/SGLang）执行推理 |
| 7 | new-api | 接收响应，统计 Token 用量，扣减余额 |
| 8 | Client | 收到 OpenAI 格式响应 |

---

## 6. 部署方案

### 6.1 Docker Compose 配置

```yaml
# docker-compose.yml
version: '3.8'

services:
  # =========================================
  # new-api 服务
  # =========================================
  new-api:
    image: calciumion/new-api:latest
    container_name: new-api
    restart: always
    ports:
      - "3000:3000"
    volumes:
      - ./new-api-data:/data
    environment:
      # 数据库（生产建议使用 MySQL/PostgreSQL）
      - SQL_DSN=root:your_password@tcp(mysql:3306)/newapi?charset=utf8mb4
      # 会话密钥（必须修改）
      - SESSION_SECRET=your_random_session_secret_here
      # 初始管理员 Token
      - INITIAL_ROOT_TOKEN=sk-admin-initial-token
      # Redis（可选，用于速率限制和缓存）
      - REDIS_CONN_STRING=redis://redis:6379
      # 日志级别
      - LOG_LEVEL=info
      # 启用语义缓存（可选）
      # - ENABLE_METRIC=true
    depends_on:
      - mysql
      - redis
    networks:
      - ai-network

  # =========================================
  # MySQL 数据库
  # =========================================
  mysql:
    image: mysql:8.0
    container_name: new-api-mysql
    restart: always
    environment:
      - MYSQL_ROOT_PASSWORD=your_password
      - MYSQL_DATABASE=newapi
    volumes:
      - ./mysql-data:/var/lib/mysql
    networks:
      - ai-network

  # =========================================
  # Redis（速率限制 + 分布式锁）
  # =========================================
  redis:
    image: redis:7-alpine
    container_name: new-api-redis
    restart: always
    networks:
      - ai-network

  # =========================================
  # GPUStack（若与 new-api 同机部署）
  # =========================================
  gpustack:
    image: gpustack/gpustack:latest
    container_name: gpustack
    restart: always
    ports:
      - "80:80"
    volumes:
      - ./gpustack-data:/var/lib/gpustack
    environment:
      - GPUSTACK_SERVER_URL=http://0.0.0.0:80
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    networks:
      - ai-network

networks:
  ai-network:
    driver: bridge
```

### 6.2 环境变量说明

| 变量 | 必填 | 说明 |
|------|------|------|
| `SQL_DSN` | 是 | 数据库连接串，默认 SQLite |
| `SESSION_SECRET` | 是 | Web 会话密钥，随机字符串 |
| `INITIAL_ROOT_TOKEN` | 否 | 初始管理员 API Token |
| `REDIS_CONN_STRING` | 否 | Redis 连接串，启用后支持分布式速率限制 |
| `BATCH_UPDATE_ENABLED` | 否 | 启用批量余额更新（高并发场景） |
| `GEMINI_SAFETY_SETTING` | 否 | Gemini 安全设置 |
| `GLOBAL_API_RATE_LIMIT` | 否 | 全局 API 速率限制（次/分钟） |

### 6.3 启动流程

```bash
# 1. 克隆 new-api
git clone https://github.com/Calcium-Ion/new-api.git
cd new-api

# 2. 修改 docker-compose.yml（填入实际密码和配置）
cp .env.example .env
vim .env

# 3. 启动服务
docker compose up -d

# 4. 访问管理界面
open http://localhost:3000
# 默认账号: root / 123456
```

---

## 7. Channel 配置指南

### 7.1 通过 Web UI 添加 GPUStack Channel

1. 登录 new-api 管理后台（`http://your-server:3000`）
2. 进入 **渠道管理** → **新建渠道**
3. 填写以下信息：

| 字段 | 值 | 说明 |
|------|-----|------|
| 渠道类型 | `OpenAI` | Type=1，兼容 GPUStack |
| 名称 | `GPUStack-Node1` | 便于识别 |
| Base URL | `http://gpustack-host:80` | GPUStack 服务地址 |
| API Key | `gsk_xxxx_xxxx` | GPUStack API Key（可多行） |
| 模型 | `Qwen2.5-72B-Instruct,Llama-3.1-8B` | 该节点上的模型，逗号分隔 |
| 优先级 | `10` | 数值越高优先级越高 |
| 权重 | `3` | 同优先级内按权重分配流量 |

### 7.2 通过 API 创建 Channel

```bash
curl -X POST http://localhost:3000/api/channel \
  -H "Authorization: Bearer sk-admin-token" \
  -H "Content-Type: application/json" \
  -d '{
    "type": 1,
    "name": "GPUStack-Node1",
    "key": "gsk_xxxx_xxxx_your_api_key",
    "base_url": "http://gpustack-host:80",
    "models": "Qwen2.5-72B-Instruct,Qwen2.5-32B-Instruct,Llama-3.1-8B",
    "model_mapping": "{\"gpt-4\": \"Qwen2.5-72B-Instruct\", \"gpt-3.5-turbo\": \"Llama-3.1-8B\"}",
    "priority": 10,
    "weight": 3,
    "status": 1
  }'
```

### 7.3 模型名称映射（Model Mapping）

GPUStack 使用真实模型名称（如 `Qwen2.5-72B-Instruct`），但很多客户端习惯使用 OpenAI 的模型名（如 `gpt-4`）。

在 Channel 的 `model_mapping` 字段中配置映射：

```json
{
  "gpt-4": "Qwen2.5-72B-Instruct",
  "gpt-4-turbo": "Qwen2.5-72B-Instruct",
  "gpt-3.5-turbo": "Llama-3.1-8B",
  "text-embedding-ada-002": "bge-m3"
}
```

**效果**：客户端发送 `model: "gpt-4"`，new-api 转发给 GPUStack 时自动替换为 `model: "Qwen2.5-72B-Instruct"`。

### 7.4 多 API Key 配置

GPUStack Channel 支持配置多个 API Key（轮询使用），在 `key` 字段中换行分隔：

```
gsk_prod_key1_xxxx
gsk_prod_key2_xxxx
gsk_prod_key3_xxxx
```

---

## 8. 用户与配额管理

### 8.1 Token（API Key）管理

在 new-api 中，每个用户/应用使用独立的 Token（`sk-xxx` 格式）：

```bash
# 创建用户 Token
curl -X POST http://localhost:3000/api/token \
  -H "Authorization: Bearer sk-admin-token" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Team-A",
    "remain_quota": 1000000,       // 1M tokens 额度
    "expired_time": -1,             // -1 = 永不过期
    "unlimited_quota": false,
    "models": "Qwen2.5-72B-Instruct,Llama-3.1-8B",  // 允许使用的模型
    "subnet": ""                    // IP 白名单（空=不限制）
  }'
```

### 8.2 Group（用户分组）

通过 Group 设置不同用户的费率倍率：

| Group | 费率倍率 | 适用场景 |
|-------|----------|----------|
| `default` | 1.0x | 普通用户 |
| `vip` | 0.8x | VIP 用户（8 折） |
| `internal` | 0.1x | 内部团队（1 折） |
| `free` | 0x | 免费试用 |

### 8.3 计费规则

new-api 按以下公式计费：

```
费用 = (Prompt Tokens × PromptRatio + Completion Tokens × CompletionRatio)
       × ModelRatio × GroupRatio / 500000
```

**ModelRatio 配置示例**（在系统设置中配置）：

```json
{
  "Qwen2.5-72B-Instruct": 4.0,
  "Qwen2.5-32B-Instruct": 2.0,
  "Llama-3.1-8B": 0.5,
  "bge-m3": 0.1
}
```

---

## 9. 高级功能

### 9.1 多 GPUStack 节点负载均衡

**场景**：两个 GPUStack 集群，一个高性能（A100），一个标准（4090），实现智能路由：

```bash
# 高性能节点（主力）
curl -X POST http://localhost:3000/api/channel \
  -H "Authorization: Bearer sk-admin-token" \
  -d '{
    "type": 1,
    "name": "GPUStack-A100",
    "key": "gsk_a100_key",
    "base_url": "http://gpustack-a100:80",
    "models": "Qwen2.5-72B-Instruct",
    "priority": 10,
    "weight": 4
  }'

# 标准节点（辅助）
curl -X POST http://localhost:3000/api/channel \
  -H "Authorization: Bearer sk-admin-token" \
  -d '{
    "type": 1,
    "name": "GPUStack-4090",
    "key": "gsk_4090_key",
    "base_url": "http://gpustack-4090:80",
    "models": "Qwen2.5-72B-Instruct,Llama-3.1-8B",
    "priority": 10,
    "weight": 1
  }'
```

**流量分配**：同优先级下，A100 节点获得 80%（4/5）流量，4090 节点 20%（1/5）。

### 9.2 故障转移（Failover）

```bash
# 主节点（Priority=10）
{ "priority": 10, "name": "GPUStack-Primary" }

# 备用节点（Priority=5，仅主节点不可用时启用）
{ "priority": 5, "name": "GPUStack-Backup" }
```

new-api 自动健康检测：Channel 连续失败后自动降级（Status=3），恢复后重新启用。

### 9.3 Channel 健康检测配置

在管理后台 → 渠道管理 中：

- **测试渠道**：手动触发健康检测
- **自动禁用**：Channel 测试失败后自动设为 `AutoDisabled` 状态
- **自动启用**：定时任务检测恢复后自动重新启用

### 9.4 流式响应（Streaming）

GPUStack 支持 SSE 流式输出，new-api 完整透传：

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Authorization: Bearer sk-user-token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-72B-Instruct",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": true
  }'
```

### 9.5 Embedding 和多模态支持

GPUStack 支持的其他端点，new-api 均可透传：

```bash
# Embedding
curl http://localhost:3000/v1/embeddings \
  -H "Authorization: Bearer sk-user-token" \
  -d '{"model": "bge-m3", "input": "hello world"}'

# 图像生成
curl http://localhost:3000/v1/images/generations \
  -H "Authorization: Bearer sk-user-token" \
  -d '{"model": "stable-diffusion-xl", "prompt": "a cat"}'

# TTS
curl http://localhost:3000/v1/audio/speech \
  -H "Authorization: Bearer sk-user-token" \
  -d '{"model": "kokoro", "input": "Hello world", "voice": "alloy"}'
```

---

## 10. 迁移路径

### 10.1 迁移前准备

- [ ] 记录当前 sub2api 的所有配置（后端地址、API key）
- [ ] 统计当前用户数量和使用模式
- [ ] 规划 new-api 的部署方式（同机 or 独立服务器）
- [ ] 准备 MySQL 数据库（生产环境不建议使用 SQLite）

### 10.2 迁移步骤

```
Phase 1: 部署 new-api（并行运行）
────────────────────────────────
1. 部署 new-api（使用新端口，如 3000）
2. 在 new-api 中添加 GPUStack Channel
3. 创建管理员 Token
4. 验证 new-api → GPUStack 链路正常

Phase 2: 用户迁移
────────────────────────────────
5. 为每个团队/用户创建 new-api Token
6. 配置各 Token 的模型权限和配额
7. 向用户提供新的 API Endpoint 和 Token
8. 设置 2 周并行期（sub2api 和 new-api 同时运行）

Phase 3: 切换
────────────────────────────────
9. 确认所有用户已切换到 new-api
10. 将流量入口（Nginx/LB）指向 new-api :3000
11. 关闭 sub2api
12. 监控 new-api 运行状态
```

### 10.3 用户侧变更（最小化影响）

用户仅需修改两处配置：

```python
# 修改前（sub2api）
client = OpenAI(
    base_url="http://sub2api-host:port/v1",
    api_key="sub2api-key"
)

# 修改后（new-api）
client = OpenAI(
    base_url="http://new-api-host:3000/v1",  # 新地址
    api_key="sk-new-user-token"               # 新 Token
)

# API 调用代码无需任何修改
response = client.chat.completions.create(
    model="Qwen2.5-72B-Instruct",
    messages=[{"role": "user", "content": "你好"}]
)
```

---

## 11. 验证方案

### 11.1 基础连通性测试

```bash
# 测试 new-api 服务
curl http://localhost:3000/api/status

# 测试模型列表
curl http://localhost:3000/v1/models \
  -H "Authorization: Bearer sk-your-token"

# 测试 Chat（非流式）
curl http://localhost:3000/v1/chat/completions \
  -H "Authorization: Bearer sk-your-token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-72B-Instruct",
    "messages": [{"role": "user", "content": "请回复 OK"}],
    "max_tokens": 10
  }'
```

### 11.2 流式响应测试

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Authorization: Bearer sk-your-token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-72B-Instruct",
    "messages": [{"role": "user", "content": "数到 10"}],
    "stream": true
  }' | grep "data:"
```

### 11.3 负载均衡验证

发送 10 次请求，检查是否路由到不同 Channel：

```bash
for i in {1..10}; do
  curl -s http://localhost:3000/v1/chat/completions \
    -H "Authorization: Bearer sk-your-token" \
    -H "Content-Type: application/json" \
    -d '{"model": "Qwen2.5-72B-Instruct", "messages": [{"role": "user", "content": "你的节点是？"}], "max_tokens": 20}' \
    | jq -r '.choices[0].message.content'
done
```

在 new-api 管理后台 → 日志 中，检查每次请求的 Channel 分配情况。

### 11.4 计费验证

```bash
# 查询 Token 余额
curl http://localhost:3000/api/user/self \
  -H "Authorization: Bearer sk-your-token"

# 发送一次请求后，再次查询余额，验证扣减正确
```

### 11.5 Channel 健康检测

```bash
# 在管理后台触发 Channel 测试
curl -X POST http://localhost:3000/api/channel/test/1 \
  -H "Authorization: Bearer sk-admin-token"

# 查询 Channel 状态
curl http://localhost:3000/api/channel?p=0 \
  -H "Authorization: Bearer sk-admin-token"
```

### 11.6 端到端验证清单

- [ ] new-api 服务正常启动，Web UI 可访问
- [ ] GPUStack Channel 添加成功，状态为"启用"
- [ ] Channel 健康检测通过（ResponseTime > 0）
- [ ] 非流式 Chat 请求正常返回
- [ ] 流式 Chat 请求正常 SSE 输出
- [ ] Embedding 请求正常返回向量
- [ ] 模型映射功能正常（`gpt-4` → `Qwen2.5-72B-Instruct`）
- [ ] Token 余额扣减正确
- [ ] 多节点负载均衡流量分配符合权重比例
- [ ] Channel 故障时自动降级到备用 Channel
- [ ] 用户 Token 模型权限限制生效

---

## 附录：GPUStack API 端点参考

GPUStack 暴露的 OpenAI-compatible 端点：

| 端点 | 方法 | 功能 |
|------|------|------|
| `/v1/models` | GET | 获取已部署模型列表 |
| `/v1/chat/completions` | POST | Chat 对话（含流式） |
| `/v1/completions` | POST | 文本补全 |
| `/v1/embeddings` | POST | 文本向量化 |
| `/v1/images/generations` | POST | 图像生成 |
| `/v1/audio/speech` | POST | TTS 文字转语音 |
| `/v1/audio/transcriptions` | POST | STT 语音转文字 |
| `/v1/messages` | POST | Anthropic-compatible（Claude 格式） |
| `/v1/rerank` | POST | Jina-compatible 重排序 |

认证方式：`Authorization: Bearer gsk_xxxx_xxxx`

GPUStack 内置轮询负载均衡：请求自动分发到所有可用 Worker 节点，无需 new-api 做额外处理。

---

*文档基于 new-api（Calcium-Ion/new-api）和 GPUStack 最新版本编写。*
