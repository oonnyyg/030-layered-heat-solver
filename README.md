# layered-heat-solver

分层一维瞬态传热求解器。仅依赖 Python 标准库（含 SQLite），通过
`python -m layered_heat --state <目录>` 启动，按 JSONL（每行一个 JSON 请求/响应）
在 stdin/stdout 上收发。

## 命令

- `import_component` `{name, layers:[{thickness, k, cv, n}]}` — 导入构件。
  截面积固定 1 m²；`thickness` 厚度 m、`k` 导热系数 W/(m·K)、`cv` 体积热容
  J/(m³·K)、`n` 等分单元数；总单元数 ≤ 5000，材料参数必须为正且有限。
- `create_case` `{name, component, dt, t_end, initial, left, right, sources?}` —
  建工况。`initial` 为标量或与单元数等长的数组（℃），尺寸不符拒绝。
  配置不可变，改配置须另建工况（重名拒绝）。
- `start` / `resume` `{case, max_steps?}` — 启动/续算；`max_steps` 用于暂停。
- `progress` `{case}` — 查进度（已提交步数、当前时刻、状态）。
- `export_temperature` / `export_energy` `{case}` — 导出当前温度场 / 能量收支；
  未完成时段以 `complete:false` 标记，不当完成结果导出。

边界 `left`/`right`：`{type:"temperature"|"flux", series:[[t,v],...]}`，
热流以向内为正（W/m²），零流即绝热。`sources` 为每层体积热源 W/m³
（标量、按层数组或按层分段序列）。所有序列时点自 0 严格递增，分段恒定，
变化点所在步起用新值。

## 离散公式

守恒有限体积，单元中心存温度。相邻单元 i、j 的界面热阻为两侧半单元热阻之和
（不做导热系数平均）：

```
R_ij = (dx_i/2)/k_i + (dx_j/2)/k_j      热流 q = (T_j - T_i) / R_ij
```

温度边界用半单元热阻 `(dx/2)/k`。时间积分采用后向欧拉，内部热流与温度边界
热流均用新时刻温度计算：

```
cv_i·V_i·(T_i^{n+1} - T_i^n)/dt = Σ_j (T_j^{n+1} - T_i^{n+1})/R_ij + q̇_i·V_i
```

形成三对角线性系统，用 Thomas 算法 O(n) 求解，不建稠密矩阵。

## 时间边界

步边界取三者并集：自零起定步长 dt 网格、全部分段变化点、终点 t_end
（末步自动截断到 t_end）。变化点被拆步，该点起的步使用新值。

## 能量收支与验证

热含量以 0 ℃ 为基准。每步记录：边界净输入（两端分别按新温度或给定热流 × dt）、
热源供热（各层 q̇·V·dt）、蓄热变化（Σ cv·V·ΔT）。逐步累计并校验

```
|累计残差| ≤ 1e-9 · max(1 J, Σ|初始单元热含量|, Σ 每步每端及每层热源换热量绝对值)
```

溢出（非有限温度）或超差时该步失败且不提交，工况标记 `failed`。

## 检查点与恢复

每步在 SQLite 事务内同时提交步记录与检查点（温度、时刻、累计收支、缩放因子）。
暂停（`max_steps`）或异常后 `resume` 从检查点继续：已提交步不重计、不从初温
重算，续算结果与一次性运行完全一致（测试覆盖）。

## 测试

```
python -m unittest discover -s tests
```

覆盖：非法/非有限输入与初温尺寸校验、单元数上限、界面热阻稳态解析解、
绝热守恒（热含量不变）、变化点拆步与新值生效、末步截断、暂停/续算一致性、
失败步不提交、未完时段不导出为完成、JSONL 端到端流程。
