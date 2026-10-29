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

        # 统计信息
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

        print(f"✓ Data is imported to: {filepath}")
        print(f"  - Sheet 'data': {len(samples)} instances")
        print(f"  - Sheet 'statistics': statistics")

    def _write_data_sheet(self, writer, samples):
        features_list = []
        labels = []

        for x, y in samples:
            features_list.append(x)
            labels.append(y + 1)

        df = pd.DataFrame(features_list)
        df['label'] = labels
        df.to_excel(writer, sheet_name='data', index=False)
        print(f"  → Scale: {df.shape[0]} rows × {df.shape[1]} columns")

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

        # 1. 类别分布统计
        stats_data.append(['=== class distribution ===', '', '', ''])
        stats_data.append(['class', 'count', 'actual proportion(%)', 'targeted proportion(%)'])

        label_counts = Counter(labels)
        total_samples = len(labels)

        for class_idx in sorted(label_counts.keys()):
            count = label_counts[class_idx]
            actual_ratio = count / total_samples * 100
            target_ratio = self.class_ratios[class_idx] * 100 if class_idx < len(self.class_ratios) else 0
            stats_data.append([
                f'Class {class_idx + 1}',
                count,
                f'{actual_ratio:.2f}',
                f'{target_ratio:.2f}'
            ])

        stats_data.append(['', '', '', ''])

        if numeric_features:
            stats_data.append(['=== statistics of continuous features ===', '', '', ''])
            stats_data.append(['feature', 'min', 'max', 'mean', 'std', 'median'])

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
            stats_data.append(['=== statistics of discrete features ===', '', '', ''])
            stats_data.append(['feature', 'number of unique values', 'distribution of values', ''])

            for feature in categorical_features:
                value_counts = df_features[feature].value_counts().sort_index()
                unique_count = len(value_counts)

                distribution = ', '.join([
                    f'{val}:({count} count, {count / total_samples * 100:.1f}%)'
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
                f'Class {class_idx + 1}',  # 显示时+1
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


class MultiStageImbalancedStream:
    def __init__(self, stream_generator, stages, seed=None, exact_distribution=True):
        self.stream_generator = stream_generator
        self.stages = stages
        self.seed = seed
        self.exact_distribution = exact_distribution

        if not stages or len(stages) == 0:
            raise ValueError("At least one stage must be specified")

        n_classes = len(stages[0]['class_ratios'])
        for i, stage in enumerate(stages):
            if len(stage['class_ratios']) != n_classes:
                raise ValueError(f"Stage {i + 1} has different number of classes")

        self.n_classes = n_classes
        self.n_stages = len(stages)

    def generate_all_stages(self):
        all_samples = []
        stage_metadata = []

        print("=" * 70)
        print(f"开始生成 {self.n_stages} 个阶段的数据")
        print("=" * 70)

        for stage_idx, stage_config in enumerate(self.stages):
            print(f"\n【阶段 {stage_idx + 1}/{self.n_stages}】")
            print(f"  样本数: {stage_config['n_samples']}")
            print(f"  类别分布: {stage_config['class_ratios']}")

            stream = self.stream_generator()

            imb_stream = MultiClassImbalancedStream(
                stream=stream,
                class_ratios=stage_config['class_ratios'],
                seed=self.seed + stage_idx if self.seed else None,
                exact_distribution=self.exact_distribution
            )

            stage_samples = list(imb_stream.take(stage_config['n_samples']))

            start_index = len(all_samples)
            end_index = start_index + len(stage_samples) - 1

            stage_meta = {
                'stage_id': stage_idx + 1,
                'n_samples': stage_config['n_samples'],
                'class_ratios': stage_config['class_ratios'],
                'start_index': start_index,
                'end_index': end_index,
                'start_row_excel': start_index + 2,
                'end_row_excel': end_index + 2,
                'statistics': imb_stream.get_statistics()
            }

            stage_metadata.append(stage_meta)
            all_samples.extend(stage_samples)

            print(f"  ✓ 完成: 生成了 {len(stage_samples)} 个样本")

        print("\n" + "=" * 70)
        print(f"✓ 所有阶段生成完成! 总样本数: {len(all_samples)}")
        print("=" * 70)

        return all_samples, stage_metadata

    def export_to_excel(self, filepath: str):
        all_samples, stage_metadata = self.generate_all_stages()

        if not all_samples:
            raise ValueError("No samples to export")

        print(f"\n正在导出数据到: {filepath}")

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            self._write_data_sheet(writer, all_samples, stage_metadata)
            self._write_multistage_statistics(writer, all_samples, stage_metadata)

        print(f"✓ 数据已成功导出到: {filepath}")
        print(f"  - Sheet 'data': {len(all_samples)} 条样本")
        print(f"  - Sheet 'stage_statistics': {self.n_stages} 个阶段的详细统计")

    def _write_data_sheet(self, writer, all_samples, stage_metadata):
        features_list = []
        labels = []
        stage_ids = []

        for stage_meta in stage_metadata:
            start_idx = stage_meta['start_index']
            end_idx = stage_meta['end_index']

            for i in range(start_idx, end_idx + 1):
                x, y = all_samples[i]
                features_list.append(x)
                labels.append(y + 1)
                stage_ids.append(stage_meta['stage_id'])

        df = pd.DataFrame(features_list)
        df.insert(0, 'stage', stage_ids)
        df['label'] = labels

        df.to_excel(writer, sheet_name='data', index=False)
        print(f"  → 数据维度: {df.shape[0]} 行 × {df.shape[1]} 列")

    def _write_multistage_statistics(self, writer, all_samples, stage_metadata):
        stats_data = []

        stats_data.append(['=== 多阶段数据流概览 ===', '', '', '', '', ''])
        stats_data.append(['总阶段数', self.n_stages, '', '', '', ''])
        stats_data.append(['总样本数', len(all_samples), '', '', '', ''])
        stats_data.append(['类别数量', self.n_classes, '', '', '', ''])
        stats_data.append(['生成模式', 'EXACT' if self.exact_distribution else 'RANDOM', '', '', '', ''])
        stats_data.append(['', '', '', '', '', ''])

        stats_data.append(['=== 各阶段详细信息 ===', '', '', '', '', ''])
        stats_data.append(['阶段', '样本数', '起始样本编号', '结束样本编号', 'Excel起始行', 'Excel结束行'])

        for stage_meta in stage_metadata:
            stats_data.append([
                f"阶段 {stage_meta['stage_id']}",
                stage_meta['n_samples'],
                stage_meta['start_index'],
                stage_meta['end_index'],
                stage_meta['start_row_excel'],
                stage_meta['end_row_excel']
            ])

        stats_data.append(['', '', '', '', '', ''])

        for stage_meta in stage_metadata:
            stats_data.append([f"=== 阶段 {stage_meta['stage_id']} - 类别分布 ===", '', '', '', '', ''])
            stats_data.append(['类别', '目标比例(%)', '实际数量', '实际比例(%)', '误差(%)', ''])

            target_ratios = stage_meta['class_ratios']
            actual_ratios = stage_meta['statistics']['actual_ratios']
            actual_counts = stage_meta['statistics']['generated_counts']

            for class_idx in range(self.n_classes):
                target = target_ratios[class_idx] * 100
                actual = actual_ratios.get(class_idx, 0) * 100
                count = actual_counts.get(class_idx, 0)
                error = abs(actual - target)

                stats_data.append([
                    f'Class {class_idx + 1}',
                    f'{target:.2f}',
                    count,
                    f'{actual:.2f}',
                    f'{error:.2f}',
                    ''
                ])

            stats_data.append(['', '', '', '', '', ''])

        stats_data.append(['=== 整体类别分布（所有阶段合并） ===', '', '', '', '', ''])
        stats_data.append(['类别', '总数量', '总比例(%)', '', '', ''])

        labels = [y + 1 for x, y in all_samples]
        label_counts = Counter(labels)
        total_samples = len(labels)

        for class_idx in sorted(label_counts.keys()):
            count = label_counts[class_idx]
            ratio = count / total_samples * 100
            stats_data.append([
                f'Class {class_idx}',
                count,
                f'{ratio:.2f}',
                '',
                '',
                ''
            ])

        stats_data.append(['', '', '', '', '', ''])

        features_list = [x for x, y in all_samples]
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

        if numeric_features:
            stats_data.append(['=== 数值特征统计（全局） ===', '', '', '', '', ''])
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
            stats_data.append(['=== 分类特征统计（全局） ===', '', '', '', '', ''])
            stats_data.append(['特征名', '唯一值数量', '取值分布', '', '', ''])

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
                    '',
                    '',
                    ''
                ])

        df_stats = pd.DataFrame(stats_data)
        df_stats.to_excel(writer, sheet_name='stage_statistics', index=False, header=False)

        print(f"  → 统计信息包含:")
        print(f"    • 阶段数量: {self.n_stages} 个")
        print(f"    • 类别分布: {self.n_classes} 个类别")
        print(f"    • 数值特征: {len(numeric_features)} 个")
        print(f"    • 分类特征: {len(categorical_features)} 个")


if __name__ == "__main__":
    from river import synth

    def create_stream():
        return synth.RandomRBF(seed_model=42,
                               seed_sample=42,
                               n_classes=5,
                               n_features=21,
                               n_centroids=10)

    stages = [
        {
            'n_samples': 40000,
            'class_ratios': [0.581395, 0.232558, 0.116279, 0.058139, 0.011629]
        },
        {
            'n_samples': 40000,
            'class_ratios': [0.011629, 0.581395, 0.232558, 0.116279, 0.058139]
        },
        {
            'n_samples': 40000,
            'class_ratios': [0.058139, 0.011629, 0.581395, 0.232558, 0.116279]
        },
        {
            'n_samples': 40000,
            'class_ratios': [0.116279, 0.058139, 0.011629, 0.581395, 0.232558]
        },
        {
            'n_samples': 40000,
            'class_ratios': [0.232558, 0.116279, 0.058139, 0.011629, 0.581395]
        }
    ]

    multi_stage_gen = MultiStageImbalancedStream(
        stream_generator=create_stream,
        stages=stages,
        seed=123,
        exact_distribution=True
    )

    multi_stage_gen.export_to_excel('dynamic_RandomRBF.xlsx')