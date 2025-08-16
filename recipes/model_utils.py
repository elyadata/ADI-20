from torch import nn
from logging import getLogger

logger = getLogger(__name__)


def validate_layer_count(model, n_layers, purpose="drop"):
    """
    Validates the number of layers specified for modification.

    Args:
        model (nn.ModuleList): The model layers to be modified.
        n_layers (int): Number of layers to drop or freeze.
        purpose (str): Purpose of validation, either 'drop' or 'freeze' for logging.

    Returns:
        int: Total number of layers in the model.

    Raises:
        ValueError: If n_layers is not in the valid range (0 < n_layers < total_layers).
    """
    total_layers = len(model)
    if not (0 < n_layers < total_layers):
        raise ValueError(
            f"n_layers ({n_layers}) for '{purpose}' must be greater than 0 and less than the total number of layers ({total_layers}).")
    return total_layers


def drop_layers(model: nn.ModuleList, n_layers: int, direction: str = "end") -> nn.ModuleList:
    """
    Drops a specified number of layers from the start or end of a model.

    Args:
        model (nn.ModuleList): The model from which layers will be dropped.
        n_layers (int): Number of layers to drop.
        direction (str): Direction from which to drop layers, either 'start' or 'end'.

    Returns:
        nn.ModuleList: The modified model with specified layers removed.

    Raises:
        TypeError: If model is not an instance of nn.ModuleList.
        ValueError: If direction is not 'start' or 'end', or if n_layers is invalid.
    """
    if not isinstance(model, nn.ModuleList):
        raise TypeError("Model must be an nn.ModuleList for dropping layers.")
    if direction not in ["start", "end"]:
        raise ValueError(f"Incorrect value for direction '{direction}' specified. Value must be in ['start', 'end'].")

    total_layers = validate_layer_count(model, n_layers, purpose="drop")
    logger.info(f"Initial number of layers: {total_layers}.")

    if direction == "start":
        deleted_indices = list(range(n_layers))
        del model[:n_layers]
    else:
        deleted_indices = list(range(total_layers - n_layers, total_layers))
        del model[-n_layers:]

    logger.info(f"Deleted layer indices: {deleted_indices}")
    logger.info(f"Updated number of layers: {len(model)}.")
    return model


def freeze_layers(model: nn.ModuleList, n_layers: int) -> nn.ModuleList:
    """
    Freezes the first n_layers layers of the model by setting requires_grad to False.

    Args:
        model (nn.ModuleList): The model containing layers to be frozen.
        n_layers (int): Number of layers to freeze from the start.

    Returns:
        nn.ModuleList: The model with specified layers frozen in place.

    Raises:
        TypeError: If model is not an instance of nn.ModuleList.
        ValueError: If n_layers is invalid.
    """
    if not isinstance(model, nn.ModuleList):
        raise TypeError("Model must be an nn.ModuleList for freezing layers.")

    total_layers = validate_layer_count(model, n_layers, purpose="freeze")
    logger.info(f"Freezing the first {n_layers} out of {total_layers} layers.")

    for i, layer in enumerate(model[:n_layers]):
        for param in layer.parameters():
            param.requires_grad = False
    logger.info(f"Layers {list(range(n_layers))} have been frozen.")

    return model
