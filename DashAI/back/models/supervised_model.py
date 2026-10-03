"""Base class for models trained and evaluated with target columns."""

import logging
import math
from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, Optional, final

from kink import di

from DashAI.back.core.enums.metrics import LevelEnum, SplitEnum
from DashAI.back.dependencies.database.models import Metric
from DashAI.back.models.base_model import BaseModel

if TYPE_CHECKING:
    from DashAI.back.dataloaders.classes.dashai_dataset import DashAIDataset

logger = logging.getLogger(__name__)


class SupervisedModel(BaseModel):
    """Base contract for supervised models in DashAI.

    Supervised models are trained with input features and target columns, and
    their metrics are computed by comparing ground-truth values against model
    predictions. The metric persistence helpers live here instead of
    ``BaseModel`` because they assume ``y_true``/``y_pred`` evaluation and
    train/validation/test splits.
    """

    @abstractmethod
    def train(
        self,
        x_train: "DashAIDataset",
        y_train: "DashAIDataset",
        x_validation: "DashAIDataset" = None,
        y_validation: "DashAIDataset" = None,
    ) -> "BaseModel":
        """Train the model with supervised input features and targets.

        Parameters
        ----------
        x_train : DashAIDataset
            The input features for training.
        y_train : DashAIDataset
            The target labels for training.
        x_validation : DashAIDataset, optional
            Input features for
            validation. Defaults to None.
        y_validation : DashAIDataset, optional
            Target labels for
            validation. Defaults to None.

        Returns
        -------
        BaseModel
            The trained model instance.
        """
        raise NotImplementedError

    @final
    def _save_metrics(
        self,
        split: SplitEnum,
        level: LevelEnum,
        results: Dict[str, float],
        log_index: int = None,
        fold_index: int = None,
        inner_fold_index: int = None,
    ):
        """Persist computed metric values to the database.

        Handles step-index computation and upsert logic for LAST-level metrics.
        Called internally by `calculate_metrics` after scores are computed.

        Parameters
        ----------
        split : SplitEnum
            The data split the metrics belong to (TRAIN,
            VALIDATION, or TEST).
        level : LevelEnum
            The granularity level (LAST, TRIAL, STEP, or
            BATCH). LAST-level entries are upserted; others are inserted.
        results : Dict[str, float]
            Mapping of metric name to score value.
        log_index : int, optional
            Explicit step index for the entries.
            If None, the next index is derived from existing database
            entries. Defaults to None.
        """
        with di["session_factory"]() as db:
            # Initialize tracking dict if not exists
            if not hasattr(self, "_metric_step_counters"):
                self._metric_step_counters = {}

            # Create a unique key for this run/split/level combination
            counter_key = (self.run_id, split, level)

            # 1. Determine log_index
            if counter_key not in self._metric_step_counters:
                steps = (
                    db.query(Metric.step)
                    .filter_by(run_id=self.run_id, split=split, level=level)
                    .order_by(Metric.step.desc())
                    .limit(2)
                    .all()
                )

                if not steps:
                    current, previous = 0, 0
                elif len(steps) == 1:
                    current, previous = steps[0][0], 0
                else:
                    current, previous = steps[0][0], steps[1][0]

                self._metric_step_counters[counter_key] = {
                    "current": current,
                    "previous": previous,
                }

            counter = self._metric_step_counters[counter_key]

            current_max = counter["current"]
            previous_max = counter["previous"]

            # Compute delta (preserve spacing)
            delta = current_max - previous_max
            if delta <= 0:
                delta = 1

            # Case 1: no log_index -> advance naturally
            if log_index is None or log_index <= current_max:
                log_index = current_max + delta

            # Update the in-memory tracker
            counter["previous"] = current_max
            counter["current"] = log_index

            # 2. Handle 'LAST' level replacement logic
            if level == LevelEnum.LAST:
                for name, value in results.items():
                    existing = (
                        db.query(Metric)
                        .filter_by(
                            run_id=self.run_id, split=split, level=level, name=name
                        )
                        .first()
                    )

                    if existing:
                        existing.value = value
                        existing.step = log_index
                    else:
                        db.add(
                            Metric(
                                run_id=self.run_id,
                                split=split,
                                level=level,
                                name=name,
                                value=value,
                                step=log_index,
                            )
                        )

            # 3. Standard logging (STEP, BATCH, TRIAL) - just insert
            else:
                metric_entries = [
                    Metric(
                        run_id=self.run_id,
                        split=split,
                        level=level,
                        name=name,
                        value=score,
                        step=log_index,
                        fold_index=fold_index,
                        inner_fold_index=inner_fold_index,
                    )
                    for name, score in results.items()
                ]
                db.add_all(metric_entries)

            db.commit()

    def _score_split(
        self,
        split: SplitEnum,
        x_data: "DashAIDataset" = None,
        y_data: "DashAIDataset" = None,
    ) -> Optional[Dict[str, float]]:
        """Score the metrics declared for a split, dropping non-finite results.

        This is the one scoring loop behind ``calculate_metrics``, which
        persists the scores, and ``compute_metrics``, which returns them. They
        used to carry a copy each and the copies drifted: only the logging one
        dropped non-finite scores, so a NaN that was too suspect to write to
        the database still reached the objective of an HPO trial through the
        other, and poisoned every comparison downstream of it.

        Parameters
        ----------
        split : SplitEnum
            The data split to evaluate (TRAIN, VALIDATION, or TEST).
        x_data : DashAIDataset, optional
            Input features. If None, the dataset stored in the model for the
            given split is used. Defaults to None.
        y_data : DashAIDataset, optional
            Target labels. If None, the labels stored in the model for the
            given split are used. Defaults to None.

        Returns
        -------
        Optional[Dict[str, float]]
            The finite scores keyed by metric name, or None when there was
            nothing to score: no metrics declared for the split, or no data
            available for it. None and an empty dict are different answers --
            the first says the question could not be asked, the second that
            every metric was asked and none returned a usable number.
        """
        metrics = getattr(self, f"{split.value}_metrics", None)

        # No metrics declared for this split: nothing to ask.
        if not metrics:
            return None

        # Load data if not provided
        if x_data is None or y_data is None:
            if self.x_data is None or self.y_data is None:
                return None
            x_data = self.x_data[split.value]
            y_data = self.y_data[split.value]

        # If data is empty after retrieval, there is nothing to score
        if x_data is None or y_data is None:
            return None

        # Make predictions and transform outputs
        y_pred = self.predict(x_data)
        y_transformed = self.prepare_output(y_data, is_fit=False)

        # Calculate metric scores
        results = {}
        for metric in metrics:
            score = metric.score(y_transformed, y_pred)
            if not math.isfinite(score):
                logger.warning(
                    "Metric %s returned a non-finite value (%s) for split %s "
                    "(e.g. only one class present in the split). Skipping.",
                    metric.__name__,
                    score,
                    split,
                )
                continue
            results[metric.__name__] = score

        return results

    @final
    def calculate_metrics(
        self,
        split: SplitEnum = SplitEnum.VALIDATION,
        level: LevelEnum = LevelEnum.LAST,
        log_index: int = None,
        x_data: "DashAIDataset" = None,
        y_data: "DashAIDataset" = None,
        fold_index: int = None,
        inner_fold_index: int = None,
    ):
        """Calculate and save metrics for a given data split and level.

        Parameters
        ----------
        split : SplitEnum
            The data split to evaluate (TRAIN, VALIDATION,
            or TEST). Defaults to SplitEnum.VALIDATION.
        level : LevelEnum
            The metric granularity level (LAST, TRIAL,
            STEP, or BATCH). Defaults to LevelEnum.LAST.
        log_index : int, optional
            Explicit step index for the metric
            entry. If None, the next step index is computed automatically.
            Defaults to None.
        x_data : DashAIDataset, optional
            Input features. If None, the
            dataset stored in the model for the given split is used.
            Defaults to None.
        y_data : DashAIDataset, optional
            Target labels. If None, the
            labels stored in the model for the given split are used.
            Defaults to None.
        """
        # If no metrics or run_id, skip calculation. The run_id is checked
        # before scoring rather than after: without a run there is nowhere to
        # persist the result, and predicting only to discard the numbers is
        # wasted work.
        if not getattr(self, f"{split.value}_metrics", None) or not self.run_id:
            return

        results = self._score_split(split, x_data=x_data, y_data=y_data)
        if results is None:
            return

        # Save to database
        self._save_metrics(
            split=split,
            level=level,
            results=results,
            log_index=log_index,
            fold_index=fold_index,
            inner_fold_index=inner_fold_index,
        )

        # Report the epoch to whoever is watching, AFTER persisting: the reporter
        # is allowed to raise (Optuna prunes that way), and the metrics of the
        # epoch that triggered the stop should survive it.
        if (
            self._epoch_reporter is not None
            and level is LevelEnum.EPOCH
            and split is SplitEnum.VALIDATION
        ):
            self._epoch_reporter(results, log_index)

    # The sibling of calculate_metrics: same scoring loop, but the scores are
    # returned instead of written to the database. Used by the CV evaluation
    # loop, where they become the objective of an HPO trial.
    def compute_metrics(
        self,
        split: SplitEnum = SplitEnum.TEST,
        x_data: "DashAIDataset" = None,
        y_data: "DashAIDataset" = None,
    ) -> Dict[str, float]:
        """Calculate and return metric scores for a given data split.

        Parameters
        ----------
        split : SplitEnum
            The data split to evaluate (TRAIN, VALIDATION,
            or TEST). Defaults to SplitEnum.VALIDATION.
        x_data : DashAIDataset, optional
            Input features. If None, the
            dataset stored in the model for the given split is used.
            Defaults to None.
        y_data : DashAIDataset, optional
            Target labels. If None, the
            labels stored in the model for the given split are used.
            Defaults to None.

        Returns
        -------
        Dict[str, float]
            A dictionary mapping metric names to their computed scores. A
            metric that scored a non-finite value is absent from the mapping
            rather than present with a NaN: see ``_score_split``. Callers that
            need a particular metric must therefore check that it is there.
        """
        # "Nothing to score" and "nothing scored finite" answer this method's
        # question the same way: no usable numbers for this split.
        return self._score_split(split, x_data=x_data, y_data=y_data) or {}
