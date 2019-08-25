# pylint: disable=missing-docstring
from ray.tune.logger import JsonLogger, CSVLogger, TFLogger
from ray.tune.result import TIMESTEPS_TOTAL, TRAINING_ITERATION
from ray.rllib.utils.annotations import override


class CustomTFLogger(TFLogger):
    @override(TFLogger)
    def on_result(self, result):
        summaries = result["info"]["learner"]["summaries"]
        del result["info"]["learner"]["summaries"]
        time = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]
        self._file_writer.add_summary(summaries, time)
        super().on_result(result)


DEFAULT_LOGGERS = (JsonLogger, CSVLogger, CustomTFLogger)
