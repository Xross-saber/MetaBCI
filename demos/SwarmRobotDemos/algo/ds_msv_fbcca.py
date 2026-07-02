"""DS-MSV-FBCCA: 动态停止多奇异值滤波器组CCA。"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy import ndarray
from scipy.signal import cheby1, sosfiltfilt
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class DSMSVFBCCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    """动态停止多奇异值滤波器组CCA分类器。

    Parameters
    ----------
    filterbank : list of ndarray
        各子带的SOS滤波器系数。
    filterweights : ndarray, optional
        子带权重。默认使用 ``(m + 1)^(-1.25) + 0.24``。
    singular_weights : sequence of float
        奇异值权重，默认复现原算法的 ``[1.4, 0.37, 0.10, 0.3, -0.1]``。
    score_power : float
        相关分数幂次，原算法为4。
    decision_thresholds : sequence of float, optional
        各动态停止时间窗的前两名分数差阈值。
    """

    def __init__(
        self,
        filterbank: List[ndarray],
        filterweights: Optional[ndarray] = None,
        singular_weights: Sequence[float] = (1.4, 0.37, 0.10, 0.3, -0.1),
        score_power: float = 4.0,
        decision_thresholds: Optional[Sequence[float]] = None,
    ) -> None:
        self.filterbank = filterbank
        self.filterweights = filterweights
        self.singular_weights = singular_weights
        self.score_power = score_power
        self.decision_thresholds = decision_thresholds

    def fit(
        self,
        X: Optional[ndarray] = None,
        y: Optional[ndarray] = None,
        Yf: Optional[ndarray] = None,
    ):
        """保存CCA参考模板；``X``和``y``仅用于兼容MetaBCI接口。"""
        if Yf is None:
            raise ValueError("必须通过Yf提供CCA参考信号")
        references = np.asarray(Yf, dtype=float)
        if references.ndim != 3:
            raise ValueError("Yf形状必须为(类别数, 2*谐波数, 采样点数)")
        if not np.all(np.isfinite(references)):
            raise ValueError("Yf包含NaN或Inf")
        if not self.filterbank:
            raise ValueError("filterbank不能为空")

        singular_weights = np.asarray(self.singular_weights, dtype=float)
        if singular_weights.ndim != 1 or singular_weights.size == 0:
            raise ValueError("singular_weights必须是一维非空序列")

        if self.filterweights is None:
            filterweights = np.asarray(
                [(index + 1) ** (-1.25) + 0.24 for index in range(len(self.filterbank))],
                dtype=float,
            )
        else:
            filterweights = np.asarray(self.filterweights, dtype=float)
        if filterweights.shape != (len(self.filterbank),):
            raise ValueError("filterweights数量必须与filterbank一致")

        self.Yf_ = references
        self.singular_weights_ = singular_weights
        self.filterweights_ = filterweights
        self._template_qr_cache = {}
        return self

    def _template_qrs(self, sample_count: int):
        if sample_count > self.Yf_.shape[-1]:
            raise ValueError(
                "输入采样点数{}超过参考模板长度{}".format(
                    sample_count, self.Yf_.shape[-1]
                )
            )
        if sample_count not in self._template_qr_cache:
            self._template_qr_cache[sample_count] = [
                np.linalg.qr(reference[:, :sample_count].T, mode="reduced")[0]
                for reference in self.Yf_
            ]
        return self._template_qr_cache[sample_count]

    def transform(self, X: ndarray) -> ndarray:
        """计算各试次对所有目标的融合分数。

        Parameters
        ----------
        X : ndarray
            EEG数据，形状为 ``(试次数, 通道数, 采样点数)``。
        """
        check_is_fitted(self, ("Yf_", "filterweights_", "singular_weights_"))
        trials = np.asarray(X, dtype=float)
        if trials.ndim == 2:
            trials = trials[np.newaxis, ...]
        if trials.ndim != 3:
            raise ValueError("X形状必须为(试次数, 通道数, 采样点数)")
        if not np.all(np.isfinite(trials)):
            raise ValueError("X包含NaN或Inf")

        sample_count = trials.shape[-1]
        template_qrs = self._template_qrs(sample_count)
        scores = np.zeros((trials.shape[0], len(template_qrs)), dtype=float)

        for subband_index, sos in enumerate(self.filterbank):
            filtered = sosfiltfilt(sos, trials, axis=-1)
            for trial_index, trial in enumerate(filtered):
                signal_qr = np.linalg.qr(trial.T, mode="reduced")[0]
                for class_index, template_qr in enumerate(template_qrs):
                    singular_values = np.linalg.svd(
                        signal_qr.T @ template_qr, compute_uv=False
                    )
                    used = min(singular_values.size, self.singular_weights_.size)
                    rho = np.dot(
                        self.singular_weights_[:used], singular_values[:used]
                    )
                    scores[trial_index, class_index] += (
                        self.filterweights_[subband_index]
                        * float(rho) ** float(self.score_power)
                    )
        return scores

    def decision_function(self, X: ndarray) -> ndarray:
        """返回各目标分数，与``transform``一致。"""
        return self.transform(X)

    def predict(self, X: ndarray) -> ndarray:
        """返回从0开始的类别编号，与MetaBCI的CCA类保持一致。"""
        return np.argmax(self.transform(X), axis=-1)

    def predict_with_confidence(self, X: ndarray) -> Tuple[ndarray, ndarray]:
        """返回类别以及第一、第二名分数差。"""
        scores = self.transform(X)
        labels = np.argmax(scores, axis=-1)
        if scores.shape[1] < 2:
            confidence = np.full(scores.shape[0], np.inf)
        else:
            sorted_scores = np.sort(scores, axis=-1)
            confidence = sorted_scores[:, -1] - sorted_scores[:, -2]
        return labels, confidence

    def should_stop(self, confidence: float, decision_index: int) -> bool:
        """依据原算法阈值判断当前时间窗是否可提前停止。"""
        if self.decision_thresholds is None:
            return False
        thresholds = np.asarray(self.decision_thresholds, dtype=float)
        if not 0 <= decision_index < thresholds.size:
            raise IndexError("decision_index超出动态停止阈值范围")
        return float(confidence) >= thresholds[decision_index]

    @staticmethod
    def generate_filterbank(
        passbands: Sequence[Sequence[float]],
        srate: float = 250.0,
        order: int = 12,
        rp: float = 0.5,
    ) -> List[ndarray]:
        """按原算法生成Chebyshev-I型SOS滤波器组。"""
        filters = []
        for low, high in passbands:
            if not 0 < low < high < float(srate) / 2:
                raise ValueError("子带({}, {})超出有效频率范围".format(low, high))
            filters.append(
                cheby1(
                    int(order),
                    float(rp),
                    [float(low), float(high)],
                    btype="bandpass",
                    output="sos",
                    fs=float(srate),
                )
            )
        return filters


DEFAULT_DECISION_WINDOWS = (0.92, 1.08, 1.16, 1.24, 1.32, 1.48)
DEFAULT_DECISION_THRESHOLDS = (
    1.4295840561689048,
    1.140595977629339,
    1.5930524437084526,
    0.9609362553967971,
    0.5073649343021734,
    -np.inf,
)
DEFAULT_PASSBANDS = (
    (6.0, 91.0),
    (14.0, 90.0),
    (22.0, 90.0),
    (30.0, 90.0),
    (38.0, 90.0),
    (46.0, 90.0),
)
