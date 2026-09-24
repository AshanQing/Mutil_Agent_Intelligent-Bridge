import json
import os
import glob
import logging
import numpy as np
import math

logger = logging.getLogger(__name__)

class SampleSelector:
    def __init__(self, few_shots_dir):
        """
        初始化样本选择器
        """
        self.samples = self._load_sample_database(few_shots_dir)
        if not self.samples:
            logger.warning(f"警告: 在 {few_shots_dir} 中未找到任何样本文件。")
        else:
            logger.info(f"样本库加载完成，共 {len(self.samples)} 个样本。")

    def select_best_k(self, design_input, k=3):
        """
        基于工程特征向量选择最相似的样本
        """
        if not self.samples:
            return []

        # 1. 硬过滤：线路类型必须一致
        # (整体式参考整体式，分离式参考分离式，这是结构设计的根本区别)
        target_type = design_input.get("线路类型", "整体式")
        filtered_samples = [s for s in self.samples if s.get("线路类型") == target_type]
        
        # 如果过滤后没有样本，降级为使用所有样本（虽然不太可能）
        candidates = filtered_samples if filtered_samples else self.samples
        
        logger.info(f"正在从 {len(candidates)} 个同类型({target_type})样本中筛选...")

        # 2. 提取特征向量
        # 目标向量
        input_vec, input_mask = self._extract_engineering_feature_payload(design_input)
        
        scored_samples = []
        for sample in candidates:
            sample_vec, sample_mask = self._extract_engineering_feature_payload(sample)
            
            # 3. 计算加权距离 (距离越小越相似)
            # 我们关注地形的匹配度远高于其他因素
            dist = self._calculate_weighted_distance(
                input_vec, sample_vec, input_mask, sample_mask
            )
            
            # 记录用于排序
            scored_samples.append({
                "sample": sample,
                "score": dist,
                "debug_vec": sample_vec
            })

        # 4. 排序 (距离升序)
        scored_samples.sort(key=lambda x: x["score"])
        
        best_samples = []
        for item in scored_samples[:k]:
            s = item["sample"]
            score = item["score"]
            vec = item["debug_vec"]
            fname = s.get('_source_file', 'unknown')
            
            # 打印调试信息，让你知道为什么选它
            logger.info(f"选中样本: {fname} (距离分: {score:.2f}) | "
                        f"特征:[Max高差={vec[0]:.1f}m, Avg高差={vec[1]:.1f}m, 构造物={vec[2]}个]")
            
            # 再次确保结构完整性（防止 Z 线丢失）
            self._validate_sample_integrity(s)
            best_samples.append(s)

        return best_samples

    def _extract_engineering_feature_payload(self, data):
        """
        提取关键工程数值特征，返回 numpy 数组
        特征维度定义:
        [0]: 最大地形高差 (反映沟谷深度，决定是否为高墩桥)
        [1]: 平均地形高差 (反映整体起伏剧烈程度)
        [2]: 构造物数量 (反映约束条件的复杂度)
        [3]: 最小平曲线半径 (反映线形指标，越小越难)
        """
        diffs = []
        terrain_valid = False

        def extract_diffs_from_line(line_data):
            if not isinstance(line_data, dict):
                return [], False

            table = line_data.get("地形特征表") or line_data.get("K线地形特征表")
            table_diffs = []
            if isinstance(table, list):
                for row in table:
                    if not isinstance(row, dict):
                        continue
                    value = next(
                        (
                            row.get(key)
                            for key in (
                                "高差（设计线-地面线）",
                                "高差(设计线-地面线)",
                                "高差",
                                "design_ground_diff",
                            )
                            if row.get(key) is not None
                        ),
                        None,
                    )
                    try:
                        table_diffs.append(float(value))
                    except (TypeError, ValueError):
                        continue
            elif isinstance(table, str):
                for line in table.splitlines():
                    if not line.lstrip().startswith("|") or "---" in line:
                        continue
                    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
                    if len(cells) < 4 or "高差" in cells[-1]:
                        continue
                    try:
                        table_diffs.append(float(cells[-1]))
                    except ValueError:
                        continue
            if table_diffs:
                return table_diffs, True

            vertical = line_data.get("纵断面结构")
            source = vertical if isinstance(vertical, dict) else line_data
            if "设计线高程序列" in source and "地形线高程序列" in source:
                try:
                    d_list = source.get("设计线高程序列", [])
                    t_list = source.get("地形线高程序列", [])
                    if d_list and t_list:
                        design_points = sorted(
                            (float(item["桩号"]), float(item["高程"]))
                            for item in d_list
                        )
                        stations = np.array([point[0] for point in design_points])
                        elevations = np.array([point[1] for point in design_points])
                        local_diffs = []
                        for ground in t_list:
                            station = float(ground["桩号"])
                            design_elevation = float(
                                np.interp(station, stations, elevations)
                            )
                            local_diffs.append(design_elevation - float(ground["高程"]))
                        return local_diffs, True
                except (KeyError, TypeError, ValueError):
                    pass
            return [], False

        # 尝试从 K/Z 中提取
        for route_key in ("K", "Z", "Z1", "Z2"):
            if route_key not in data:
                continue
            route_diffs, route_valid = extract_diffs_from_line(data[route_key])
            diffs.extend(route_diffs)
            terrain_valid = terrain_valid or route_valid
        
        # 如果实在没有数据（比如纯构造物输入），给默认值
        if not diffs:
            max_diff = 0
            avg_diff = 0
        else:
            # 只关心“正高差”（即桥墩高度），负高差（挖方）对设桥影响较小（直接断开）
            # 所以我们过滤出 > 0 的部分来计算特征
            positive_diffs = [d for d in diffs if d > 0]
            if positive_diffs:
                max_diff = max(positive_diffs)
                avg_diff = np.mean(positive_diffs)
            else:
                max_diff = 0
                avg_diff = 0

        # --- 2. 构造物特征 ---
        struct_count = len(data.get("构造物信息") or data.get("障碍物信息") or [])

        # --- 3. 线形特征 ---
        # 寻找最小半径
        min_radius = 99999.0
        radius_valid = False
        # 辅助查找
        def find_min_r(line_key):
            r_val = 99999.0
            nonlocal radius_valid
            if line_key in data and data[line_key]:
                curves = data[line_key].get("平曲线结构", [])
                for c in curves:
                    # 兼容 '半径R' 或 'R'
                    r = c.get("半径R") or c.get("R")
                    if r and isinstance(r, (int, float)):
                        r_val = min(r_val, r)
                        radius_valid = True
            return r_val

        min_radius = min(find_min_r("K"), find_min_r("Z"))
        if min_radius > 50000: min_radius = 5000.0 # 归一化大半径

        vector = np.array([max_diff, avg_diff, struct_count, min_radius], dtype=float)
        validity = np.array([terrain_valid, terrain_valid, True, radius_valid], dtype=bool)
        return vector, validity

    def _extract_engineering_features(self, data):
        """兼容旧调用方，只返回数值向量。"""
        vector, _ = self._extract_engineering_feature_payload(data)
        return vector

    def _calculate_weighted_distance(self, vec1, vec2, mask1=None, mask2=None):
        """
        计算加权距离。
        不同的特征在相似性判断中权重不同。
        """
        # 权重定义：
        # 最大高差 (40%): 最决定性的因素，30m墩和80m墩的设计完全不同
        # 平均高差 (30%): 决定是连续桥梁还是路桥相间
        # 构造物数量 (20%): 决定布跨的受限程度
        # 最小半径 (10%): 影响较小，除非是超小半径
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        
        # 数据归一化处理（简单的比例差异）
        # 避免 2000(半径) 淹没 30(高度) 的影响
        # 算法：abs(v1 - v2) / (max(v1, v2) + epsilon)
        # 这样计算的是“相对差异率”
        
        if mask1 is None:
            mask1 = np.ones(len(vec1), dtype=bool)
        if mask2 is None:
            mask2 = np.ones(len(vec2), dtype=bool)
        diffs = []
        for i in range(len(vec1)):
            if not bool(mask1[i]) or not bool(mask2[i]):
                diffs.append(1.0)
                continue
            v1 = vec1[i]
            v2 = vec2[i]
            denominator = max(abs(v1), abs(v2))
            if denominator < 1e-6: # 避免除以0
                d = 0
            else:
                d = abs(v1 - v2) / denominator
            diffs.append(d)
            
        weighted_dist = np.dot(np.array(diffs), weights)
        return weighted_dist

    def _load_sample_database(self, path):
        """内部方法：加载目录下所有 JSON，并标准化结构 (Flatten)"""
        samples = []
        if not os.path.exists(path):
            return samples

        search_pattern = os.path.join(path, "*.json")
        files = glob.glob(search_pattern)

        for fp in sorted(files):
            try:
                with open(fp, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 兼容 list 或 dict
                items = data if isinstance(data, list) else [data]
                
                for item in items:
                    if isinstance(item, dict):
                        # === 结构标准化 (Flatten) ===
                        # 如果样本包含 'input' 键，将其内容提升到根目录
                        if 'input' in item and isinstance(item['input'], dict):
                            input_data = item.pop('input') 
                            item.update(input_data)
                        
                        item["_source_file"] = os.path.basename(fp)
                        samples.append(item)
                        
            except Exception as e:
                logger.warning(f"无法读取样本文件 {fp}: {e}")
        
        return samples

    def _validate_sample_integrity(self, sample):
        """验证样本完整性"""
        line_type = sample.get("线路类型", "")
        source_file = sample.get("_source_file", "unknown")
        
        if line_type == "分离式":
            has_k = "K" in sample or "K线" in sample
            has_z = any(k.startswith("Z") for k in sample.keys())
            if not has_z:
                logger.warning(f"⚠️  警告: 样本 {source_file} (分离式) 缺失 Z 线数据，相似度计算可能不准。")

    # _extract_features 和 _simplify_line_data 方法不再需要用于计算相似度，
    # 但为了兼容可能得接口调用，或者如果需要存日志，可以保留，或者直接删除。
    # 这里为了代码整洁，建议删除原有的 _extract_features 文本提取方法。
