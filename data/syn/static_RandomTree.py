import numpy as np
import pandas as pd
from river import base
from river.utils.skmultiflow_utils import check_random_state
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional


class MultiClassImbalancedStream:
    """
    多分类不平衡数据流生成器，支持精确类别分布控制

    改进：使用预先计算的精确数量，确保类别分布严格符合指定比例

    Parameters
    ----------
    stream : river数据生成器
        原始数据流
    class_ratios : list of float
        每个类别的比例，总和应为1.0
    seed : int, optional
        随机种子
    exact_distribution : bool, default=True
        是否使用精确分布（True=精确，False=随机抽样）
    max_attempts : int, optional
        防止无限循环的最大尝试次数
    """

    def __init__(self, stream, class_ratios, seed=None, exact_distribution=True, max_attempts=10000):
        self.stream = stream
        self.class_ratios = np.array(class_ratios)
        self.seed = seed
        self.rng = check_random_state(seed)
        self.exact_distribution = exact_distribution
        self.max_attempts = max_attempts

        # 验证class_ratios
        if not np.isclose(np.sum(self.class_ratios), 1.0, atol=1e-3):
            raise ValueError(
                f"class_ratios must sum to 1.0, got {np.sum(self.class_ratios)}"
            )

        if np.any((self.class_ratios < 0) | (self.class_ratios > 1)):
            raise ValueError("All class_ratios must be between 0.0 and 1.0")

        self.n_classes = len(class_ratios)
        self.stream_iter = iter(self.stream)

        # 统计信息
        self.generated_counts = Counter()
        self.discarded_counts = Counter()
        self.total_generated = 0

        # 精确分布模式的额外变量
        self.class_schedule = []  # 预先计算的类别序列
        self.schedule_index = 0  # 当前调度索引

    def __iter__(self):
        return self

    def __next__(self):
        if self.exact_distribution:
            return self._next_exact()
        else:
            return self._next_random()

    def _next_exact(self):
        """精确分布模式：从预计算的类别序列中获取目标类别"""
        # 如果调度序列为空或已用完，抛出异常
        if self.schedule_index >= len(self.class_schedule):
            raise StopIteration("Exact distribution schedule exhausted. Use take(n) to generate samples.")

        target_class = self.class_schedule[self.schedule_index]
        self.schedule_index += 1

        # 从原始流中寻找目标类别的样本
        attempts = 0
        while attempts < self.max_attempts:
            try:
                x, y = next(self.stream_iter)
                attempts += 1

                if y == target_class:
                    self.generated_counts[y] += 1
                    self.total_generated += 1
                    return x, y
                else:
                    self.discarded_counts[y] += 1

            except StopIteration:
                if hasattr(self.stream, '__iter__'):
                    self.stream_iter = iter(self.stream)
                else:
                    raise StopIteration("Stream exhausted and cannot be restarted")

        raise RuntimeError(f"Failed to find class {target_class} after {self.max_attempts} attempts")

    def _next_random(self):
        """随机分布模式：原始实现"""
        attempts = 0
        while attempts < self.max_attempts:
            target_class = self._select_target_class()

            while attempts < self.max_attempts:
                try:
                    x, y = next(self.stream_iter)
                    attempts += 1

                    if y == target_class:
                        self.generated_counts[y] += 1
                        self.total_generated += 1
                        return x, y
                    else:
                        self.discarded_counts[y] += 1

                except StopIteration:
                    if hasattr(self.stream, '__iter__'):
                        self.stream_iter = iter(self.stream)
                    else:
                        raise StopIteration("Stream exhausted and cannot be restarted")

        raise RuntimeError(f"Failed to generate instance after {self.max_attempts} attempts")

    def _select_target_class(self):
        """随机选择目标类别"""
        p = self.rng.rand()
        cumsum = 0.0

        for i, ratio in enumerate(self.class_ratios):
            cumsum += ratio
            if p < cumsum:
                return i

        return self.n_classes - 1

    def _prepare_exact_schedule(self, n):
        """
        准备精确的类别调度序列

        Parameters
        ----------
        n : int
            总样本数

        Returns
        -------
        list
            打乱后的类别序列，确保精确分布
        """
        # 计算每个类别的精确数量
        class_counts = np.round(self.class_ratios * n).astype(int)

        # 处理舍入误差：调整总和到n
        diff = n - class_counts.sum()
        if diff != 0:
            # 找到比例最大的类别进行调整
            adjust_class = np.argmax(self.class_ratios)
            class_counts[adjust_class] += diff

        # 创建类别序列
        schedule = []
        for class_idx, count in enumerate(class_counts):
            schedule.extend([class_idx] * count)

        # 打乱序列（保持随机性但总数精确）
        self.rng.shuffle(schedule)

        return schedule

    def take(self, n):
        """
        生成n个样本（精确分布模式）

        Parameters
        ----------
        n : int
            要生成的样本数量

        Yields
        ------
        tuple
            (x, y) 样本和标签
        """
        if self.exact_distribution:
            # 准备精确的类别调度
            self.class_schedule = self._prepare_exact_schedule(n)
            self.schedule_index = 0

            print(f"📊 精确分布模式:")
            for class_idx, count in enumerate(Counter(self.class_schedule).items()):
                print(f"  Class {class_idx}: {count[1]} 样本 ({count[1] / n * 100:.2f}%)")

        for _ in range(n):
            yield next(self)

    def get_statistics(self):
        """获取生成统计信息"""
        total_discarded = sum(self.discarded_counts.values())
        total_processed = self.total_generated + total_discarded

        actual_ratios = {}
        if self.total_generated > 0:
            for i in range(self.n_classes):
                actual_ratios[i] = self.generated_counts.get(i, 0) / self.total_generated

        return {
            'mode': 'exact' if self.exact_distribution else 'random',
            'target_ratios': dict(enumerate(self.class_ratios)),
            'actual_ratios': actual_ratios,
            'generated_counts': dict(self.generated_counts),
            'discarded_counts': dict(self.discarded_counts),
            'total_generated': self.total_generated,
            'total_discarded': total_discarded,
            'acceptance_rate': self.total_generated / total_processed if total_processed > 0 else 0
        }

    def export_to_excel(self, filepath: str, samples: List[Tuple[dict, int]]):
        """导出数据到Excel文件（保持原有功能）"""
        if not samples:
            raise ValueError("No samples to export")

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            self._write_data_sheet(writer, samples)
            self._write_statistics_sheet(writer, samples)

        print(f"✓ 数据已成功导出到: {filepath}")
        print(f"  - Sheet 'data': {len(samples)} 条样本")
        print(f"  - Sheet 'statistics': 详细统计信息")

    def _write_data_sheet(self, writer, samples):
        """写入数据sheet"""
        features_list = []
        labels = []

        for x, y in samples:
            features_list.append(x)
            labels.append(y + 1)  # 类别编号+1，从1开始

        df = pd.DataFrame(features_list)
        df['label'] = labels
        df.to_excel(writer, sheet_name='data', index=False)
        print(f"  → 数据维度: {df.shape[0]} 行 × {df.shape[1]} 列")

    def _write_statistics_sheet(self, writer, samples):
        """写入统计信息sheet"""
        features_list = [x for x, y in samples]
        labels = [y for x, y in samples]
        df_features = pd.DataFrame(features_list)

        numeric_features = []
        categorical_features = []

        for col in df_features.columns:
            if df_features[col].dtype in ['int64', 'float64']:
                unique_vals = df_features[col].nunique()
                if unique_vals <= 10:
                    categorical_features.append(col)
                else:
                    numeric_features.append(col)
            else:
                categorical_features.append(col)

        stats_data = []

        # 1. 类别分布统计
        stats_data.append(['=== 类别分布统计 ===', '', '', ''])
        stats_data.append(['类别', '数量', '实际比例(%)', '目标比例(%)'])

        label_counts = Counter(labels)
        total_samples = len(labels)

        for class_idx in sorted(label_counts.keys()):
            count = label_counts[class_idx]
            actual_ratio = count / total_samples * 100
            target_ratio = self.class_ratios[class_idx] * 100 if class_idx < len(self.class_ratios) else 0
            stats_data.append([
                f'Class {class_idx}',
                count,
                f'{actual_ratio:.2f}',
                f'{target_ratio:.2f}'
            ])

        stats_data.append(['', '', '', ''])

        # 2. 数值特征统计
        if numeric_features:
            stats_data.append(['=== 数值特征统计 ===', '', '', ''])
            stats_data.append(['特征名', '最小值', '最大值', '平均值', '标准差', '中位数'])

            for feature in numeric_features:
                values = df_features[feature]
                stats_data.append([
                    feature,
                    f'{values.min():.4f}',
                    f'{values.max():.4f}',
                    f'{values.mean():.4f}',
                    f'{values.std():.4f}',
                    f'{values.median():.4f}'
                ])

            stats_data.append(['', '', '', '', '', ''])

        # 3. 分类特征统计
        if categorical_features:
            stats_data.append(['=== 分类特征统计 ===', '', '', ''])
            stats_data.append(['特征名', '唯一值数量', '取值分布', ''])

            for feature in categorical_features:
                value_counts = df_features[feature].value_counts().sort_index()
                unique_count = len(value_counts)

                distribution = ', '.join([
                    f'{val}:({count}次, {count / total_samples * 100:.1f}%)'
                    for val, count in value_counts.items()
                ])

                stats_data.append([
                    feature,
                    unique_count,
                    distribution,
                    ''
                ])

            stats_data.append(['', '', '', ''])

        # 4. 生成过程统计
        stats_data.append(['=== 生成过程统计 ===', '', '', ''])
        gen_stats = self.get_statistics()

        stats_data.append(['生成模式', gen_stats['mode'].upper(), '', ''])
        stats_data.append(['总生成样本数', gen_stats['total_generated'], '', ''])
        stats_data.append(['总丢弃样本数', gen_stats['total_discarded'], '', ''])
        stats_data.append(['接受率', f"{gen_stats['acceptance_rate'] * 100:.2f}%", '', ''])
        stats_data.append(['', '', '', ''])

        stats_data.append(['各类别生成统计', '', '', ''])
        stats_data.append(['类别', '生成数量', '丢弃数量', ''])

        for class_idx in range(self.n_classes):
            gen_count = gen_stats['generated_counts'].get(class_idx, 0)
            dis_count = gen_stats['discarded_counts'].get(class_idx, 0)
            stats_data.append([
                f'Class {class_idx}',
                gen_count,
                dis_count,
                ''
            ])

        df_stats = pd.DataFrame(stats_data)
        df_stats.to_excel(writer, sheet_name='statistics', index=False, header=False)

        print(f"  → 统计信息包含:")
        print(f"    • 类别分布: {len(label_counts)} 个类别")
        print(f"    • 数值特征: {len(numeric_features)} 个")
        print(f"    • 分类特征: {len(categorical_features)} 个")


# ============================================================================
# 使用示例：对比精确模式 vs 随机模式
# ============================================================================

if __name__ == "__main__":
    from river import synth

    # 创建原始流
    original_stream = synth.RandomTree(
        seed_tree=42,
        seed_sample=42,
        n_classes=5,
        n_num_features=20,
        n_cat_features=10,
        n_categories_per_feature=4,
        max_tree_depth=3
    )

    stream_exact = MultiClassImbalancedStream(
        stream=original_stream,
        class_ratios=[0.581395, 0.232558, 0.116279, 0.058139, 0.011629],
        seed=123,
        exact_distribution=True  # 启用精确分布
    )

    samples_exact = list(stream_exact.take(200000))
    stream_exact.export_to_excel('syn-\static_RandomTree.xlsx', samples_exact)

    stats_exact = stream_exact.get_statistics()
    print(f"\n实际类别分布:")
    for cls, ratio in sorted(stats_exact['actual_ratios'].items()):
        target = stats_exact['target_ratios'][cls]
        print(f"  Class {cls}: {ratio * 100:.2f}% (目标: {target * 100:.2f}%, 误差: {abs(ratio - target) * 100:.2f}%)")