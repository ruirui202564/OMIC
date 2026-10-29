import numpy as np
import pandas as pd
from river import base
from river.utils.skmultiflow_utils import check_random_state
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional


class MultiClassImbalancedStream:
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

        self.generated_counts = Counter()
        self.discarded_counts = Counter()
        self.total_generated = 0

        self.class_schedule = []
        self.schedule_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.exact_distribution:
            return self._next_exact()
        else:
            return self._next_random()

    def _next_exact(self):
        if self.schedule_index >= len(self.class_schedule):
            raise StopIteration("Exact distribution schedule exhausted. Use take(n) to generate samples.")

        target_class = self.class_schedule[self.schedule_index]
        self.schedule_index += 1

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
        p = self.rng.rand()
        cumsum = 0.0

        for i, ratio in enumerate(self.class_ratios):
            cumsum += ratio
            if p < cumsum:
                return i

        return self.n_classes - 1

    def _prepare_exact_schedule(self, n):
        class_counts = np.round(self.class_ratios * n).astype(int)

        diff = n - class_counts.sum()
        if diff != 0:
            adjust_class = np.argmax(self.class_ratios)
            class_counts[adjust_class] += diff

        schedule = []
        for class_idx, count in enumerate(class_counts):
            schedule.extend([class_idx] * count)

        self.rng.shuffle(schedule)

        return schedule

    def take(self, n):
        if self.exact_distribution:
            self.class_schedule = self._prepare_exact_schedule(n)
            self.schedule_index = 0

            print(f"📊 精确分布模式:")
            for class_idx, count in enumerate(Counter(self.class_schedule).items()):
                print(f"  Class {class_idx}: {count[1]} 样本 ({count[1] / n * 100:.2f}%)")

        for _ in range(n):
            yield next(self)

    def get_statistics(self):
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
        if not samples:
            raise ValueError("No samples to export")

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            self._write_data_sheet(writer, samples)
            self._write_statistics_sheet(writer, samples)

        print(f"✓ 数据已成功导出到: {filepath}")
        print(f"  - Sheet 'data': {len(samples)} 条样本")
        print(f"  - Sheet 'statistics': 详细统计信息")

    def _write_data_sheet(self, writer, samples):
        features_list = []
        labels = []

        for x, y in samples:
            features_list.append(x)
            labels.append(y + 1)

        df = pd.DataFrame(features_list)
        df['label'] = labels
        df.to_excel(writer, sheet_name='data', index=False)
        print(f"  → 数据维度: {df.shape[0]} 行 × {df.shape[1]} 列")

    def _write_statistics_sheet(self, writer, samples):
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


if __name__ == "__main__":
    from river import synth

    original_stream = synth.RandomRBF(seed_model=42,
                                   seed_sample=42,
                                   n_classes=5,
                                   n_features=21,
                                   n_centroids=10)

    stream_exact = MultiClassImbalancedStream(
        stream=original_stream,
        class_ratios=[0.581395, 0.232558, 0.116279, 0.058139, 0.011629],
        seed=123,
        exact_distribution=True
    )

    samples_exact = list(stream_exact.take(200000))
    stream_exact.export_to_excel('static_RandomRBF.xlsx', samples_exact)

    stats_exact = stream_exact.get_statistics()
    print(f"\n实际类别分布:")
    for cls, ratio in sorted(stats_exact['actual_ratios'].items()):
        target = stats_exact['target_ratios'][cls]
        print(f"  Class {cls}: {ratio * 100:.2f}% (目标: {target * 100:.2f}%, 误差: {abs(ratio - target) * 100:.2f}%)")