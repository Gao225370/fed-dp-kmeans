from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from typing import Tuple, Optional

from pfl.algorithm.base import FederatedAlgorithm
from pfl.common_types import Population
from pfl.context import CentralContext
from pfl.data.dataset import Dataset, AbstractDatasetType
from pfl.hyperparam.base import AlgorithmHyperParamsType, ModelHyperParamsType, AlgorithmHyperParams
from pfl.metrics import Metrics, Weighted, MetricName, MetricValue, StringMetricName
from pfl.model.base import ModelType
from pfl.stats import MappedVectorStatistics, StatisticsType

from utils import project_subspace


@dataclass(frozen=True)
class FederatedClusterInitHyperParams(AlgorithmHyperParams):
    K: int
    center_init_send_sums_and_counts: bool
    server_dataset: Dataset
    num_iterations_svd: int
    num_iterations_weighting: int
    num_iterations_center_init: int
    multiplicative_margin: float
    minimum_server_point_weight: float
    train_cohort_size: int
    val_cohort_size: Optional[int]
    datapoint_privacy: Optional[bool] = False
    outer_product_data_clipping_bound: Optional[float] = 1.
    # --- 新增开始 ---
    # 为HyperParams对象添加新字段，以便在算法中访问
    counting_method: Optional[str] = 'original'
    num_bins: Optional[int] = 128
    # --- 新增结束 ---


class FederatedOuterProduct(FederatedAlgorithm):
    def __init__(self):
        super().__init__()

    def process_aggregated_statistics(
            self, central_context: CentralContext[AlgorithmHyperParamsType,
                                                  ModelHyperParamsType],
            aggregate_metrics: Metrics, model: ModelType,
            statistics: StatisticsType) -> Tuple[ModelType, Metrics]:
        # print(statistics['outer_product'])
        return model.apply_model_update(statistics)

    # --- 修改开始 (完全替换此方法) ---
    def simulate_one_user(
        self, model: ModelType, user_dataset: AbstractDatasetType,
        central_context: CentralContext[AlgorithmHyperParamsType,
                                        ModelHyperParamsType]
    ) -> Tuple[Optional[StatisticsType], Metrics]:
        def name_formatting_fn(s: str):
            return MetricName(s, central_context.population)

        if central_context.population == Population.TRAIN:
            statistics = MappedVectorStatistics()
            metrics = Metrics()
            algo_params = central_context.algorithm_params
            X = user_dataset.raw_data[0]
            if central_context.current_central_iteration < algo_params.num_iterations_svd:
                # Return data outer product
                clipped_X = X
                if algo_params.datapoint_privacy:
                    norms = np.sqrt((X**2).sum(axis=1))
                    clipping_mask = norms > algo_params.outer_product_data_clipping_bound
                    clipped_X[clipping_mask] = (X[clipping_mask] / norms[clipping_mask].reshape(-1, 1)) * algo_params.outer_product_data_clipping_bound
                    clipping_metrics = Weighted(np.sum(clipping_mask), len(X))
                    metrics[StringMetricName('Fraction of clipped points')] = clipping_metrics

                assert np.all(clipped_X == X)

                statistics['outer_product'] = clipped_X.transpose().dot(clipped_X)

            return statistics, metrics

        else:
            return MappedVectorStatistics(), model.evaluate(user_dataset, name_formatting_fn)

    def get_next_central_contexts(
        self,
        model: ModelType,
        iteration: int,
        algorithm_params: AlgorithmHyperParamsType,
        model_train_params: ModelHyperParamsType,
        model_eval_params: Optional[ModelHyperParamsType] = None,
    ) -> Tuple[Optional[Tuple[CentralContext[
            AlgorithmHyperParamsType, ModelHyperParamsType], ...]], ModelType,
               Metrics]:

        X_server = algorithm_params.server_dataset.raw_data[0]
        if iteration == algorithm_params.num_iterations_svd:
            # Compute singular vectors
            outer_product = model.accumulated_statistics['outer_product']
            _, V = eigh(outer_product,
                        subset_by_index=(len(outer_product) - algorithm_params.K, len(outer_product) - 1))
            model.singular_vectors = V.T

            # Project server data
            model.proj_X_server = project_subspace(model.singular_vectors, X_server)

            # --- 新增开始 ---
            # 如果使用直方图方法，则在此处提前定义桶
            if algorithm_params.counting_method == 'histogram':
                print("INFO: Defining histogram bins on the server using K-Means...")
                bin_definer = KMeans(n_clusters=algorithm_params.num_bins,
                                     n_init='auto', random_state=42)
                bin_definer.fit(model.proj_X_server)
                # 将桶中心存储在 model 对象上，以便后续广播和使用
                model.bin_centers = bin_definer.cluster_centers_
            # --- 新增结束 ---

            return None, model, Metrics()

        context = CentralContext(
                current_central_iteration=iteration,
                do_evaluation=False,
                cohort_size=algorithm_params.train_cohort_size,
                population=Population.TRAIN,
                model_train_params=model_train_params.static_clone(),
                model_eval_params=model_eval_params.static_clone(),
                algorithm_params=algorithm_params.static_clone(),
                seed=self._get_seed())

        return tuple([context]), model, Metrics()


class FederatedServerPointWeighting(FederatedAlgorithm):
    def __init__(self):
        super().__init__()

    def process_aggregated_statistics(
            self, central_context: CentralContext[AlgorithmHyperParamsType,
                                                  ModelHyperParamsType],
            aggregate_metrics: Metrics, model: ModelType,
            statistics: StatisticsType) -> Tuple[ModelType, Metrics]:
        return model.apply_model_update(statistics)

    def simulate_one_user(
            self, model: ModelType, user_dataset: AbstractDatasetType,
            central_context: CentralContext[AlgorithmHyperParamsType,
            ModelHyperParamsType]
    ) -> Tuple[Optional[StatisticsType], Metrics]:

        def name_formatting_fn(s: str):
            return MetricName(s, central_context.population)

        if central_context.population == Population.TRAIN:
            statistics = MappedVectorStatistics()
            X = user_dataset.raw_data[0]
            V = model.singular_vectors
            assert V is not None
            proj_X = project_subspace(V, X)
            algo_params = central_context.algorithm_params

            if algo_params.counting_method == 'histogram':
                # 新逻辑: 计算直方图
                assert hasattr(model, 'bin_centers'), "Bin centers not found in model!"
                bin_centers = model.bin_centers

                # 为每个客户端点找到最近的桶
                dist_matrix = pairwise_distances(proj_X, bin_centers)
                assignments = np.argmin(dist_matrix, axis=1)

                # 创建并填充直方图
                histogram = np.zeros(len(bin_centers))
                unique, counts = np.unique(assignments, return_counts=True)
                histogram[unique] = counts

                statistics['histogram'] = histogram
            else:
                # 保留原始逻辑
                assert model.proj_X_server is not None
                dist_matrix = pairwise_distances(proj_X, model.proj_X_server)
                closest_server_points = np.argmin(dist_matrix, axis=1)
                assert len(closest_server_points) == len(X)
                server_point_weights = np.zeros(len(model.proj_X_server))
                unique, counts = np.unique(closest_server_points, return_counts=True)
                server_point_weights[unique] = counts
                statistics['server_point_weights'] = server_point_weights

            return statistics, Metrics()
        else:
            return MappedVectorStatistics(), model.evaluate(user_dataset, name_formatting_fn)

    def get_next_central_contexts(
            self,
            model: ModelType,
            iteration: int,
            algorithm_params: AlgorithmHyperParamsType,
            model_train_params: ModelHyperParamsType,
            model_eval_params: Optional[ModelHyperParamsType] = None,
    ) -> Tuple[Optional[Tuple[CentralContext[
        AlgorithmHyperParamsType, ModelHyperParamsType], ...]], ModelType,
    Metrics]:

        if iteration == algorithm_params.num_iterations_weighting:
            # 1. 获取聚合后的带噪统计数据
            if algorithm_params.counting_method == 'histogram':
                print("INFO: Processing aggregated histograms...")
                noisy_stats = model.accumulated_statistics['histogram']
            else:
                noisy_stats = model.accumulated_statistics['server_point_weights']

            # 2. 计算服务器点权重
            if algorithm_params.counting_method == 'histogram':
                assert hasattr(model, 'bin_centers'), "Bin centers not found in model!"
                dist_matrix = pairwise_distances(model.proj_X_server, model.bin_centers)
                server_point_assignments = np.argmin(dist_matrix, axis=1)
                sample_weight = noisy_stats[server_point_assignments]
            else:
                sample_weight = noisy_stats

            # 3. 公共后续逻辑
            sample_weight = np.maximum(0, sample_weight)
            server_point_mask = sample_weight >= algorithm_params.minimum_server_point_weight

            # 4. 关键修复：为不同的方法使用不同的 KMeans 初始化
            if algorithm_params.counting_method == 'histogram':
                # 对于我们的新方法，保持可复现性
                server_point_clustering = KMeans(n_clusters=algorithm_params.K, n_init='auto', random_state=42)
            else:
                # 对于原始方法，完全复现原始行为（不设置随机种子）
                server_point_clustering = KMeans(n_clusters=algorithm_params.K, n_init='auto')

            # 5. 执行聚类
            if np.sum(server_point_mask) >= algorithm_params.K:
                server_point_clustering.fit(
                    model.proj_X_server[server_point_mask],
                    sample_weight=sample_weight[server_point_mask]
                )
                model.proj_server_point_centers = server_point_clustering.cluster_centers_
            else:
                print(
                    f"WARNING: Only {np.sum(server_point_mask)} server points remain after weighting... Falling back to random init.")
                indices = np.random.choice(len(model.proj_X_server), algorithm_params.K, replace=False)
                model.proj_server_point_centers = model.proj_X_server[indices]

            return None, model, Metrics()

        # ... 函数余下部分保持不变 ...
        context = CentralContext(
            current_central_iteration=iteration,
            do_evaluation=False,
            cohort_size=algorithm_params.train_cohort_size,
            population=Population.TRAIN,
            model_train_params=model_train_params.static_clone(),
            model_eval_params=model_eval_params.static_clone(),
            algorithm_params=algorithm_params.static_clone(),
            seed=self._get_seed())

        return tuple([context]), model, Metrics()
    # --- 修改结束 ---
    # --- 修改结束 ---

    # --- 修改开始 (完全替换此方法) ---
    # def get_next_central_contexts(
    #         self,
    #         model: ModelType,
    #         iteration: int,
    #         algorithm_params: AlgorithmHyperParamsType,
    #         model_train_params: ModelHyperParamsType,
    #         model_eval_params: Optional[ModelHyperParamsType] = None,
    # ) -> Tuple[Optional[Tuple[CentralContext[
    #     AlgorithmHyperParamsType, ModelHyperParamsType], ...]], ModelType,
    # Metrics]:
    #
    #     if iteration == algorithm_params.num_iterations_weighting:
    #         sample_weight = None  # 初始化
    #         if algorithm_params.counting_method == 'histogram':
    #             # 新逻辑: 处理聚合后的直方图
    #             print("INFO: Aggregating histograms and mapping back to weights...")
    #             # 隐私机制会自动对 statistics['histogram'] 加噪
    #             noisy_histogram = model.accumulated_statistics['histogram']
    #
    #             # 将桶的计数值映射回服务器点的权重
    #             assert hasattr(model, 'bin_centers'), "Bin centers not found in model!"
    #             dist_matrix = pairwise_distances(model.proj_X_server, model.bin_centers)
    #             server_point_assignments = np.argmin(dist_matrix, axis=1)
    #
    #             sample_weight = noisy_histogram[server_point_assignments]
    #         else:
    #             # 原始逻辑
    #             sample_weight = model.accumulated_statistics['server_point_weights']
    #
    #         # 公共后续逻辑
    #         sample_weight = np.maximum(0, sample_weight)  # 避免负权重
    #         server_point_mask = sample_weight >= algorithm_params.minimum_server_point_weight
    #
    #         # Cluster weighted server data
    #         server_point_clustering = KMeans(n_clusters=algorithm_params.K, n_init='auto', random_state=42)
    #
    #         # 确保有足够的点用于聚类
    #         if np.sum(server_point_mask) >= algorithm_params.K:
    #             server_point_clustering.fit(
    #                 model.proj_X_server[server_point_mask],
    #                 sample_weight=sample_weight[server_point_mask]
    #             )
    #             model.proj_server_point_centers = server_point_clustering.cluster_centers_
    #         else:
    #             print(
    #                 f"WARNING: Only {np.sum(server_point_mask)} server points remain after weighting, which is less than K={algorithm_params.K}. Falling back to random initialization.")
    #             indices = np.random.choice(len(model.proj_X_server), algorithm_params.K, replace=False)
    #             model.proj_server_point_centers = model.proj_X_server[indices]
    #
    #         return None, model, Metrics()
    #
    #     context = CentralContext(
    #         current_central_iteration=iteration,
    #         do_evaluation=False,
    #         cohort_size=algorithm_params.train_cohort_size,
    #         population=Population.TRAIN,
    #         model_train_params=model_train_params.static_clone(),
    #         model_eval_params=model_eval_params.static_clone(),
    #         algorithm_params=algorithm_params.static_clone(),
    #         seed=self._get_seed())
    #
    #     return tuple([context]), model, Metrics()
    # # --- 修改结束 ---


class FederatedInitFromProjectedCenters(FederatedAlgorithm):
    def __init__(self):
        super().__init__()

    def process_aggregated_statistics(
            self, central_context: CentralContext[AlgorithmHyperParamsType,
                                                  ModelHyperParamsType],
            aggregate_metrics: Metrics, model: ModelType,
            statistics: StatisticsType) -> Tuple[ModelType, Metrics]:

        return model.apply_model_update(statistics)

    def simulate_one_user(
        self, model: ModelType, user_dataset: AbstractDatasetType,
        central_context: CentralContext[AlgorithmHyperParamsType,
                                        ModelHyperParamsType]
    ) -> Tuple[Optional[StatisticsType], Metrics]:

        def name_formatting_fn(s: str):
            return MetricName(s, central_context.population)

        algo_params = central_context.algorithm_params
        X = user_dataset.raw_data[0]
        if central_context.population == Population.TRAIN:
            statistics = MappedVectorStatistics()
            # Compute centers in original space by proximity in projected space
            V = model.singular_vectors
            proj_X = project_subspace(V, X)
            assert model.proj_server_point_centers is not None
            dist_matrix = pairwise_distances(proj_X, model.proj_server_point_centers)

            center_assignments = -np.ones(len(proj_X), dtype=int)  # assignment stays -1 if condition not satisfied
            smallest_two_distances = np.partition(dist_matrix, 1, axis=1)[:, :2]
            assert smallest_two_distances.shape[0] == len(proj_X)
            assert smallest_two_distances.shape[1] == 2
            assignment_mask = (smallest_two_distances[:, 0] <=
                               (algo_params.multiplicative_margin * smallest_two_distances[:, 1]))
            center_assignments[assignment_mask] = np.argmin(dist_matrix, axis=1)[assignment_mask]

            point_sums = []
            point_counts = []
            for k in range(algo_params.K):
                kth_mask = center_assignments == k
                point_sums.append(X[kth_mask].sum(axis=0))
                point_counts.append(kth_mask.sum())

            point_sums = np.vstack(point_sums)
            point_counts = np.hstack(point_counts)
            if algo_params.center_init_send_sums_and_counts:
                statistics['sum_points_per_component'] = point_sums
                statistics['num_points_per_component'] = point_counts
            else:
                statistics['contributed_components'] = (point_counts > 0).astype(int)
                point_counts[point_counts == 0] = 1
                statistics['mean_points_per_component'] = point_sums / point_counts.reshape(-1, 1)

            return statistics, Metrics()
        else:
            return MappedVectorStatistics(), model.evaluate(user_dataset, name_formatting_fn)

    def get_next_central_contexts(
        self,
        model: ModelType,
        iteration: int,
        algorithm_params: AlgorithmHyperParamsType,
        model_train_params: ModelHyperParamsType,
        model_eval_params: Optional[ModelHyperParamsType] = None,
    ) -> Tuple[Optional[Tuple[CentralContext[
            AlgorithmHyperParamsType, ModelHyperParamsType], ...]], ModelType,
               Metrics]:

        if iteration == algorithm_params.num_iterations_center_init:
            # Compute initial centers in original space
            model.compute_centers()
            return None, model, Metrics()

        do_evaluation = False

        configs = []
        if not do_evaluation:
            configs.append(CentralContext(
                current_central_iteration=iteration,
                do_evaluation=False,
                cohort_size=algorithm_params.train_cohort_size,
                population=Population.TRAIN,
                model_train_params=model_train_params.static_clone(),
                model_eval_params=model_eval_params.static_clone(),
                algorithm_params=algorithm_params.static_clone(),
                seed=self._get_seed()))

        elif do_evaluation and algorithm_params.val_cohort_size is not None:
            configs.append(
                CentralContext(
                    current_central_iteration=iteration,
                    do_evaluation=do_evaluation,
                    cohort_size=algorithm_params.val_cohort_size,
                    population=Population.VAL,
                    algorithm_params=algorithm_params.static_clone(),
                    model_train_params=model_train_params.static_clone(),
                    model_eval_params=model_eval_params.static_clone(),
                    seed=self._get_seed()))
        else:
            return None, model, Metrics()

        return tuple(configs), model, Metrics()


