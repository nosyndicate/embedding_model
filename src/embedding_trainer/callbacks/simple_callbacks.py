import math

from embedding_trainer.callbacks.registry import CALLBACK_REGISTRY
from embedding_trainer.core import BaseCallback
from embedding_trainer.core.base_trainer import BaseTrainer


@CALLBACK_REGISTRY.register("print_loss")
class PrintLossCallback(BaseCallback):
    def __init__(self, log_every: int = 1) -> None:
        self.log_every = max(log_every, 1)

    def on_train_begin(self, trainer: BaseTrainer) -> None:
        print("Starting training ...")

    def on_step_end(self, trainer: BaseTrainer, step: int, loss: float) -> None:
        if step % self.log_every == 0:
            ppl = math.exp(min(loss, 100))
            print(f"step={step} loss={loss:.4f} ppl={ppl:.2f}")

    def on_evaluate_end(self, trainer: BaseTrainer, metrics: dict[str, float]) -> None:
        eval_loss = metrics.get("eval_loss", float("nan"))
        ppl = metrics.get("perplexity", float("nan"))
        step = getattr(trainer, "global_step", 0)
        print(f"[eval] step={step} eval_loss={eval_loss:.4f} ppl={ppl:.2f}")

    def on_train_end(self, trainer: BaseTrainer) -> None:
        print("Training finished.")
