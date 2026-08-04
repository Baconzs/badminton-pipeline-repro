# 鹰眼落地事件：开源方案对照

本地“鹰眼辅助复核”是单机位二维复核，不等同于正式比赛 Hawk-Eye。调研的共同边界是：先确认 bounce/landing 事件，再把该事件帧映射到球场；不能把最后一个可见点、插值点或画面外点当作地面落点。

## 参考项目

| 项目 | 可借鉴内容 | 本项目的使用边界 |
|---|---|---|
| [TrackNet-V3-based-Badminton](https://github.com/ZSHYC/TrackNet-V3-based-Badminton) | TrackNet 轨迹、规则高召回候选、BounceNet 的 `landing / hit / none` 精筛；速度骤降后检查未来活动，否决假停顿 | 仓库没有随代码提供可直接使用的 BounceNet 权重；当前先实现无模型的事件门控，不宣称已接入训练模型 |
| [TrackNetV3](https://github.com/qaz812345/TrackNetV3) | 只修补内部缺口；球从画面边缘消失或没有后续真实观测时保持 missing，不做长外推 | 终局缺球不再生成 `terminal_endpoint_proxy` |
| [MonoTrack](https://github.com/jhwang7628/monotrack) | 6 点相机/球场几何、3D 物理轨迹拟合、击球事件时序约束 | Adobe Research License 仅适合非商业研究；论文还明确排除难以标注的最后一拍地面冲击，不能直接当作本产品权重 |
| [tennis-vision](https://github.com/philippdubach/tennis-vision) / [tennis-tracking](https://github.com/ArtLabss/tennis-tracking) | 速度方向变化、短缺口处理、bounce 分类后才投影 | 网球模型和转播域不能直接用于羽毛球判罚；仅借鉴事件证据结构 |
| [tennis-vision-analysis](https://github.com/Baconzs/tennis-vision-analysis) | RANSAC/重投影质量控制的球场标定思路 | 只改善标定，不能证明羽球已经触地 |

上述羽球事件仓库采用 MIT 许可证的部分应保留版权声明；没有许可证的项目或仅提供网盘权重的项目没有被复制、打包或默认为可商用。正式引入模型前仍需确认权重、数据集和部署许可。

## 当前实现的事件门槛

`web_app/hawkeye.py` 现在要求：

- 至少 5 个真实 `model`/`classical` 观测，跨度至少约 0.16 秒；
- 最后 3 点在图像中稳定，且前面有连续运动、速度下降和方向连续性；
- 速度骤降后不能立即恢复高速活动，也不能是跨检测缺口的新分支；
- `ShotEndFrame` 已指向下一次接触，或后续存在高置信未确认击球时，直接不判落地；
- 只有通过上述门槛的原始点才进入单应性投影和 IN/OUT 计算。

当终局球被球员短暂遮挡、TrackNet 在落地后停止输出，而原视频仍能看到一个新出现的静止羽球时，
系统会再做一次独立的视频恢复：用击球前的多段背景中值与击球后的时序差分寻找小而紧凑、位置稳定的白色目标，
同时排除场线、广告和人体短暂残差。恢复结果只发布边界、`OUT/REVIEW` 和一个距离范围（不输出厘米级伪精度），
并限制在标定边线外 1 m 内；距线 20 cm 内仍降级为 `REVIEW`。

因此“没有落地证据”是一个有效结果，不是错误。对于无法在原视频中恢复静止球的遮挡、出画或人体锁定片段，
系统会保留视频复核入口，但不会输出坐标或厘米级伪精度。

## 0579a694b155 回归结论

该任务的最后硬击球为第 856 帧，`ShotEndFrame=926`，第 926 帧是远端球员回收时的未确认人体误锁。第 905–911 帧的点落在腿/袜子上，
但原视频从约第 912 帧开始在远端底线附近出现一个新的静止白色羽球，并持续到视频末尾。恢复结果为：

- `OUT`，远端底线外约 `21–48 cm`（以范围中值约 `34 cm` 做判定）；
- 证据帧约 44 帧，方法为 `visible_terminal_rest_after_occlusion`；
- 不再输出原先由人体误锁产生的 `171.6/172.8 cm`。

这是单机位辅助复核，不是正式比赛判罚；实际使用仍应回看原视频确认。
