from seqio import FeatureConverter
import tensorflow as tf
from typing import Mapping, Sequence, Optional
from seqio import utils

def non_padding_position(
    tensor: tf.Tensor, dtype: tf.dtypes.DType = tf.int32, pad_id: int = 0
) -> tf.Tensor:
  """Return a tensor with 1 on non-padding and 0 on padding positions."""
  return tf.cast(tf.not_equal(tensor, pad_id), dtype=dtype)


def _check_exact_match(
    expected_features: Sequence[str],
    actual_features: Sequence[str],
    expected_feature_source: str,
    actual_feature_source: str,
) -> None:
  """Check whether expected and actual features match one-to-one."""
  expected_features = set(expected_features)
  actual_features = set(actual_features)

  if expected_features != actual_features:
    if actual_features - expected_features:
      extra_features = actual_features - expected_features
      raise ValueError(
          f"The {actual_feature_source} contains extra features not specified "
          f"in the {expected_feature_source}: {extra_features}"
      )
    else:
      missing_features = expected_features - actual_features
      raise ValueError(
          f"The {actual_feature_source} is missing features specified "
          f"in the {expected_feature_source}: {missing_features}"
      )


def _shift_right_by_one(tensor: tf.Tensor, bos_id: int = 0) -> tf.Tensor:
  """Shift the input tensor to the right by one position without wrapping."""

  if not (tensor.dtype.is_integer or tensor.dtype.is_floating):
    raise ValueError(f"Only numeric types are supported. Got: {tensor.dtype}")
  # tf.roll wraps around the axis.
  rolled = tf.roll(tensor, shift=1, axis=0)

  # Zero out the first position by multiplying with [0, 1, 1, ..., 1].
  depth = tf.shape(tensor)[0]
  mask = tf.one_hot(0, depth=depth, on_value=0, off_value=1, dtype=tensor.dtype)

  # Expand dims of mask to broadcast to rolled.
  dim_expansion = [slice(None, None)] + [None] * (len(rolled.shape) - 1)
  mask = mask[dim_expansion]
  return rolled * mask + (1 - mask) * bos_id


def make_autoregressive_inputs(
    targets: tf.Tensor,
    sequence_id: tf.Tensor = None,
    output_dtype: Optional[tf.dtypes.DType] = None,
    bos_id: int = 0,
) -> tf.Tensor:
  """Generate inputs for an autoregressive model, by shifting the targets.
  Modified from mesh_tensorflow.transformer.transformer.autoregressive_inputs.
  For the first element of each sequence, the returned input id is 0.
  For a "packed" dataset, also pass the sequence_id tensor, which aligns
  with the targets tensor and contains different values for different
  concatenated examples.
  Example for a packed dataset:
  ```
        targets = [3, 8, 1, 9, 1, 5, 4, 1, 0, 0]
    sequence_id = [1, 1, 1, 2, 2, 3, 3, 3, 0, 0]
         inputs = [0, 3, 8, 0, 9, 0, 5, 4, 0, 0]
                            |     |        |
                            These positions are set to 0 if sequence_id is not
                            None.
  ```
  Args:
    targets: a tf.int32 tensor with shape [length].
    sequence_id: an optional tensor with the same shape as targets.
    output_dtype: an optional output data type.
    bos_id: bos id.
  Returns:
    a tensor with dtype tf.int32 and the same shape as targets.
  """
  output_dtype = output_dtype or targets.dtype
  if sequence_id is not None and not sequence_id.dtype.is_integer:
    raise ValueError(
        "The sequence_id should be integer-valued tensors for a packed dataset."
    )
  if sequence_id is not None and len(targets.shape) > 1:
    raise ValueError(
        "Only 1-D sequences are supported with packing. Got a "
        f"packed {len(targets.shape)}-D sequence."
    )

  inputs = _shift_right_by_one(targets, bos_id)
  if inputs.dtype != output_dtype:
    inputs = tf.cast(inputs, output_dtype)

  # We should have a 0 at the beginning of each sequence rather than the
  # shifted EOS (e.g. 1) from the previous sequence.
  if sequence_id is not None:
    not_first_in_sequence = tf.equal(
        sequence_id, _shift_right_by_one(sequence_id)
    )
    not_first_in_sequence = tf.cast(not_first_in_sequence, output_dtype)
    first_ids = tf.cast((1 - not_first_in_sequence) * bos_id, output_dtype)
    inputs = inputs * not_first_in_sequence + first_ids
  return inputs


class LMFeatureConverter(FeatureConverter):
  """Feature converter for a language model (decoder-only) architecture.
  The input dataset must have "targets" field only.
  One common usecase is to pre-train a decoder-only model with the standard
  language modeling objective (i.e., predict the next token given the previous
  ones) on a unlabeled text corpus which only has "targets". Then the
  pre-trained model can be fine-tuned on a supervised task, e.g., machine
  translation by concatenating "inputs" and "targets". For this use case,
  pre-train with LMFeatureConverter and fine-tune with PrefixLMFeatureConverter.
  Example: a packed dataset.
    ds = [{"targets": [3, 9, 1]}, {"targets": [4, 1]}]
    input_lengths = {"targets": 6}
    converted_ds = {
        "decoder_target_tokens": [3, 9, 1, 4, 1, 0],
         "decoder_input_tokens": [0, 3, 9, 0, 4, 0],
         "decoder_loss_weights": [1, 1, 1, 1, 1, 0],
            "decoder_positions": [0, 1, 2, 0, 1, 0],
          "decoder_segment_ids": [1, 1, 1, 2, 2, 0]
    }
  Note that two examples are packed together into one example.
  """

  TASK_FEATURES = {"targets": FeatureConverter.FeatureSpec(dtype=tf.int32)}
  MODEL_FEATURES = {
      "tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "labels": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "attention_mask": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "loss_mask": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "position_ids": FeatureConverter.FeatureSpec(dtype=tf.int32),
      # "decoder_input_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
      # "decoder_loss_weights": FeatureConverter.FeatureSpec(dtype=tf.int32),
  }
  PACKING_FEATURE_DTYPES = {
      "decoder_segment_ids": tf.int32,
      "position_ids": tf.int32,
  }

  def _convert_example(
      self, features: Mapping[str, tf.Tensor]
  ) -> Mapping[str, tf.Tensor]:
    """Convert an LM example into an example with model features."""
    # targets_segment_id is present only for a packed dataset.
    decoder_input_tokens = make_autoregressive_inputs(
        features["targets"],
        sequence_id=features.get("targets_segment_ids", None),
        bos_id=self.bos_id,
    )

    d = {
        "labels": features["targets"],
        "tokens": decoder_input_tokens,
        "loss_mask": non_padding_position(features["targets"]),
        "attention_mask": tf.reshape(tf.cast(tf.experimental.numpy.triu(tf.ones((features["targets"].shape[-1], features["targets"].shape[-1]))), dtype=tf.int32), (-1,))
    }

    if self.pack:
      d["decoder_segment_ids"] = features["targets_segment_ids"]
      d["position_ids"] = features["targets_positions"]

    return d

  def _convert_features(
      self, ds: tf.data.Dataset, task_feature_lengths: Mapping[str, int]
  ) -> tf.data.Dataset:
    """Convert the dataset to be fed to a language model."""
    ds = self._pack_or_pad(ds, task_feature_lengths)
    return ds.map(
        self._convert_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

  def get_model_feature_lengths(
      self, task_feature_lengths: Mapping[str, int]
  ) -> Mapping[str, int]:
    """Define the length relationship between task and model features."""
    decoder_length = task_feature_lengths["targets"]
    model_feature_lengths = {
        "tokens": decoder_length,
        "labels": decoder_length,
        "loss_mask": decoder_length,
        "attention_mask": decoder_length * decoder_length
    }
    if self.pack:
      model_feature_lengths["decoder_segment_ids"] = decoder_length
      model_feature_lengths["position_ids"] = decoder_length

    return model_feature_lengths