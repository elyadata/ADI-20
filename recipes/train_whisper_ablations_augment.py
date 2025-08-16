import logging
import os
import sys

import speechbrain as sb
import torch
from hyperpyyaml import load_hyperpyyaml
from metrics_utils import (generate_metrics_report, make_dir,
                           save_predictions_to_json)
from model_utils import drop_layers, freeze_layers

"""Recipe for training a DID system on ADI-17 or ADI-19 and their variants.

To run this recipe, do the following:
> python train_whisper.py hparams/train_whisper.yaml

Author
------
 * Haroun Elleuch
"""

logger = logging.getLogger(__name__)


# Brain class for dialect ID training
class DID(sb.Brain):

    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""       
        batch = batch.to(self.device)
        wavs, lens = batch.sig
        
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "augmenter"):
            wavs, lens = self.hparams.augmenter(x=wavs, lengths=lens)
            
        embeddings = self.modules.whisper(wavs, lens)

        # Embeddings must be pooled (either max, avg, std or attention) across the time-step dimension
        if hparams["pooling"] == "avg":
            pooled_embeddings = self.hparams.avg_pool(x=embeddings, lengths=None).squeeze(1)
        elif hparams["pooling"] == "std":
            pooled_embeddings = self.hparams.std_pool(x=embeddings, lengths=lens)
        elif hparams["pooling"] == "max":
            pooled_embeddings, _ = embeddings.max(dim=1)
        elif hparams["pooling"] == "attention":
            pooled_embeddings = self.modules.attention_pooling(embeddings)
        else:
            raise ValueError("Please specify a valid value for the pooling scheme: avg, std, max or attention.")

        outputs = self.modules.output_mlp(pooled_embeddings)
        outputs = self.hparams.log_softmax(outputs)

        return outputs, lens

    def compute_objectives(self, inputs, batch, stage):
        """Computes the loss given the predicted and targeted outputs.

        Arguments
        ---------
        inputs : tensors
            The output tensors from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """

        predictions, lens = inputs
        targets = batch.dialect_encoded.data

        loss = self.hparams.compute_cost(predictions, targets.squeeze(1))

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, targets, lens)

            scores, indexes = torch.max(predictions, dim=-1)
            predicted_labels = self.dialect_encoder.decode_torch(indexes)

            scores = [round(score.exp().tolist(), 5) for score in scores]
            indexes = [index.tolist() for index in indexes]
            predicted_labels = [text_lab for text_lab in predicted_labels]
            target_labels = [self.dialect_encoder.decode_torch(target)[0] for target in targets]

            prediction_dict = {
                    "scores": scores,
                    "indexes": indexes,
                    "predicted_labels": predicted_labels,
                    "target_labels": target_labels,
            }
            self.prediction_log.append(prediction_dict)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )
        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()
            self.prediction_log = []

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
                "error": self.error_metrics.summarize("average"),
            }

        # At the end of validation...
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stats["error"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            (
                old_lr_whisper,
                new_lr_whisper,
            ) = self.hparams.lr_annealing_whisper(stats["error"])
            sb.nnet.schedulers.update_learning_rate(
                self.whisper_optimizer, new_lr_whisper
            )

            # Runs on main process only
            save_predictions_to_json(
                metrics_log_dir=self.hparams.metrics_log_dir,
                filename=f"classifications_epoch_{epoch}.json",
                predictions=self.prediction_log
            )

            # Runs on main process only
            classification_scores = generate_metrics_report(
                predictions=self.prediction_log,
                plot_matrix=True,
                plot_path=os.path.join(self.hparams.metrics_log_dir, f"confusion_matrix_epoch_{epoch}.png"),
                report_path=os.path.join(self.hparams.metrics_log_dir, f"metrics_report_epoch_{epoch}.json"),
                verbose=False,
                return_dict=True
            )

            # If main process, add classification scores to stats:
            if classification_scores:
                stats.update(classification_scores)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr, "whisper_lr": old_lr_whisper},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(
                            meta=stats,
                            max_keys=["macro_f1", "weighted_f1"],
                            num_to_keep=2,
                            name=f"epoch_{epoch}")

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

            save_predictions_to_json(
                metrics_log_dir=self.hparams.metrics_log_dir,
                filename="classifications_test.json",
                predictions=self.prediction_log
            )
            generate_metrics_report(
                predictions=self.prediction_log,
                plot_matrix=True,
                plot_path=os.path.join(self.hparams.metrics_log_dir, "confusion_matrix_test.png"),
                report_path=os.path.join(self.hparams.metrics_log_dir, "metrics_report_test.json"),
                verbose=True,
                return_dict=False,
                fewer_eval_classes=hparams.get("fewer_eval_classes", False)
            )

    def init_optimizers(self):
        "Initializes the whisper optimizer and model optimizer"
        self.whisper_optimizer = self.hparams.whisper_opt_class(
            self.modules.whisper.parameters()
        )
        self.optimizer = self.hparams.opt_class(self.hparams.model.parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "whisper_opt", self.whisper_optimizer
            )
            self.checkpointer.add_recoverable("optimizer", self.optimizer)

        self.optimizers_dict = {
            "model_optimizer": self.optimizer,
            "whisper_optimizer": self.whisper_optimizer,
        }


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    We expect `prepare_common_dialect` to have been called before this,
    so that the `train.csv`, `dev.csv`,  and `test.csv` manifest files
    are available.

    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "dev" that correspond
        to the appropriate DynamicItemDataset object.
    """

    # Initialization of the label encoder. The label encoder assigns to each
    # of the observed label a unique index (e.g, 'dial01': 0, 'dial02': 1, ..)
    dialect_encoder = sb.dataio.encoder.CategoricalEncoder()
    dialect_encoder.expect_len(hparams["n_languages"])

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav", "start", "end", "duration")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop, duration):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""
        start = int(float(start))
        stop = int(float(stop))
        if float(duration) > 30:
            stop = start + (30 * 16_000)  # Used for long evaluation segments. Assuming 16 kHz audio.
        
        try:
            sig = sb.dataio.dataio.read_audio(
                {"start": start, "stop": stop, "file": wav}
            )
        except :
            import wave
            with wave.open(wav, 'rb') as wav_file:
                num_frames = wav_file.getnframes()
            logger.warning(f"Error reading file: {wav}. Start: {start}, stop: {stop}. Total wav duration: {num_frames}")
            if num_frames < stop:
                logger.warning(f"File {wav} has fewer frames than expected. Trimming segment.")
                stop = num_frames
                sig = sb.dataio.dataio.read_audio({"start": start, "stop": stop, "file": wav})
            else:
                raise ValueError(f"Error reading file: {wav}. Start: {start}, stop: {stop}. Total wav duration: {num_frames}")
        # Check if the signal is not mono
        if sig.ndim > 1 and sig.shape[0] > 1:  # sig.shape[0] is the channel dimension
            print(f"Converting stereo to mono for file: {wav}")
            sig = sig.mean(dim=0)  # Average across channels to convert to mono
        return sig

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("dialect")
    @sb.utils.data_pipeline.provides("dialect", "dialect_encoded")
    def label_pipeline(dialect):
        yield dialect
        dialect_encoded = dialect_encoder.encode_label_torch(dialect)
        yield dialect_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    for dataset in ["train", "dev", "test"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams[f"{dataset}_csv"],
            replacements=None,
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "dialect_encoded"],
        )

    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mapping.
    dialect_encoder_file = os.path.join(
        hparams["save_folder"], "dialect_encoder.txt"
    )
    dialect_encoder.load_or_create(
        path=dialect_encoder_file,
        from_didatasets=[datasets["train"]],
        output_key="dialect",
    )

    return datasets, dialect_encoder


# Recipe begins!
if __name__ == "__main__":
    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(  # metrics dir
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    make_dir(hparams["metrics_log_dir"])

    # Data preparation, to be run on only one process.
    # TODO: integrate data preparation script in recipe.

    # Create dataset objects "train", "dev", and "test" and dialect_encoder
    datasets, dialect_encoder = dataio_prep(hparams)

    # Initialize the Brain object to prepare for mask training.
    did_brain = DID(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    did_brain.dialect_encoder = dialect_encoder

    if hparams.get("number_layers_to_drop", None) is not None:
        logger.info("Dropping layers.")
        did_brain.modules.whisper.model.encoder.layers = drop_layers(
            did_brain.modules.whisper.model.encoder.layers,
            n_layers=hparams["number_layers_to_drop"],
            direction=getattr(hparams, "layer_drop_direction", "end")
        )

    if hparams.get("number_layers_to_freeze", None) is not None:
        logger.info("Freezing layer.")
        did_brain.modules.whisper.model.encoder.layers = freeze_layers(
            did_brain.modules.whisper.model.encoder.layers,
            n_layers=hparams["number_layers_to_freeze"]
        )

        logger.info("Statistics after layer update:")
        did_brain.print_trainable_parameters()
    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    did_brain.fit(
        epoch_counter=did_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["dev"],
        train_loader_kwargs=hparams["train_dataloader_options"],
        valid_loader_kwargs=hparams["test_dataloader_options"],
    )

    # Load the best checkpoint for evaluation
    test_stats = did_brain.evaluate(
        test_set=datasets["test"],
        min_key="error",
        test_loader_kwargs=hparams["test_dataloader_options"],
    )
