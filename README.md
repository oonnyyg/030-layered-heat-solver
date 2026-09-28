# layered-heat-solver

一维分层瞬态传热求解器。仅使用 Python 标准库，状态持久化在 SQLite。

## 运行

```
python -m layered_heat --state <状态目录>
```

从 stdin 逐行读 JSON 命令（JSONL），逐行向 stdout 写 JSON 响应。
`--state` 指定状态目录（默认当前目录），其中保存 `layered_heat.db`。

## 命令

- `import_component`：`{"cmd":"import_component","component":{"layers":[{"thickness":0.1,"k":1.0,"cv":2e6,"n":4}, ...]}}`
  每层给厚度(m)、导热系数(W/(m·K))、体积热容(J/(m³·K))、等分单元数。返回 `component_id`。
- `create_case`：`{"cmd":"create_case","component_id":1,"case":{...}}`，返回 `case_id`。工况字段：
  - `t_end`、`dt`（秒，末步自动截断到 `t_end`）
  - `initial_temp`：标量或长度等于总单元数的数组（℃）
  - `boundaries.left/right`：`{"type":"temperature","points":[[t,v],...]}` 或
    `{"type":"flux","points":[[t,v],...]}`（热流以向内为正，零即绝热）
  - `sources`：每层一个 `[[t,q],...]`，体积热源 W/m³，缺省为 0
  - 所有 `points` 自 `t=0` 起时间严格递增、分段恒定；变化点不得晚于 `t_end`
- `start` / `resume`：`{"cmd":"start","case_id":1,"max_steps":null}`。`max_steps`
  可用于分段推进（暂停）；`resume` 从最近检查点续算，已提交步不重算。
- `progress`：状态、当前时刻、步数、累计能量收支。
- `export_temperatures` / `export_energy`：仅导出已提交步；未完成工况
  `completed=false` 且带 `exported_through`，未完时段不会当作完成结果。

修改配置须新建工况（构件与工况创建后不可变）。

## 离散公式

截面积 1 m²，温度存于单元中心。守恒有限体积 + 后向欧拉：

```
cv_i·V_i·(T_i^{n+1} − T_i^n)/Δt = Σ_j G_ij·(T_j^{n+1} − T_i^{n+1}) + 边界项 + q_i·V_i
```

- 界面热阻 `R_ij = (dx_i/2)/k_i + (dx_j/2)/k_j`，`G_ij = 1/(R_ij·A)`；
  不对导热系数做算术/调和平均。
- 温度边界用半单元热阻 `R = (dx/2)/k`，热流 `q = (T_b − T_1)/R`（用新时刻温度）。
- 热流边界直接以给定值（向内为正）进入右端项；零流即绝热。
- 时间网格 = {0, k·dt ≤ t_end} ∪ 全部分段变化点 ∪ {t_end} 的并集；
  变化点处拆步，自该点起的步使用新值（在步起点取值）。
- 三对角系数矩阵用 Thomas 算法 O(n) 求解，不建稠密矩阵。

## 能量收支

热含量基准 0 ℃。每步记录：两端边界净输入、各层热源供热、蓄热变化，
残差 `r = 蓄热变化 − (边界输入 + 热源供热)`。累计校核：

```
|Σr| ≤ 1e-9 · max(1 J, Σ|初始单元热含量|, Σ 每步|各端热流·dt| 与 |各层热源·V·dt|)
```

任一步出现数值溢出/非有限值或收支超差：该步不提交，工况标记 `failed`，须另建工况。

## 检查点与恢复

每个已提交步与其检查点（温度、时刻、累计收支、分母）在同一 SQLite 事务中写入；
暂停或异常后 `resume` 从最近完整检查点恢复，已提交步不重计、不从初温重算。

## 验证

`python -m unittest discover -s tests`，覆盖：

- 非法/非有限材料参数、单元数上限、初温尺寸、时点单调性的拒绝；
- 双端绝热 + 均匀热源的解析升温与零残差守恒；
- 双层串联热阻稳态温度分布与解析解一致；
- 分段恒定变化点拆步且新值自该点生效；
- 末步截断到 `t_end`；
- 暂停/续算与一次性运行逐步结果完全一致；
- 未完成工况导出不标记完成；累计收支残差 ≤ 1e-9 容差。
