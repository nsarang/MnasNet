"""Predefined MnasNet models."""


import tensorflow as tf
import re
from collections import namedtuple
from MnasNet import MnasNetModel


BlockArgs = namedtuple(
    "BlockArgs",
    [
        "kernel_size",
        "num_repeat",
        "input_filters",
        "output_filters",
        "expand_ratio",
        "id_skip",
        "strides",
        "se_ratio",
    ],
    defaults=(None,) * 8,
)


GlobalParams = namedtuple(
    "GlobalParams",
    [
        "batch_norm_momentum",
        "batch_norm_epsilon",
        "dropout_rate",
        "data_format",
        "input_shape",
        "num_classes",
        "depth_multiplier",
        "depth_divisor",
        "min_depth",
        "stem_size",
        "normalize_input",
    ],
    defaults=(None,) * 11,
)



def decode_block_string(block_string):
    """Gets a MNasNet block through a string notation of arguments.
    E.g. r2_k3_s2_e1_i32_o16_se0.25_noskip: r - number of repeat blocks,
    k - kernel size, s - strides (1-9), e - expansion ratio, i - input filters,
    o - output filters, se - squeeze/excitation ratio
    Args:
      block_string: a string, a string representation of block arguments.
    Returns:
      A BlockArgs instance.
    Raises:
      ValueError: if the strides option is not correctly specified.
    """

    assert isinstance(block_string, str)
    ops = block_string.split("_")
    options = {}
    for op in ops:
        splits = re.split(r"(\d.*)", op)
        if len(splits) >= 2:
            (key, value) = splits[:2]
            options[key] = value

    if "s" not in options or len(options["s"]) != 2:
        raise ValueError("Strides options should be a pair of integers.")

    return BlockArgs(
        kernel_size=int(options["k"]),
        num_repeat=int(options["r"]),
        input_filters=int(options["i"]),
        output_filters=int(options["o"]),
        expand_ratio=int(options["e"]),
        id_skip="noskip" not in block_string,
        se_ratio=(float(options["se"]) if "se" in options else None),
        strides=[int(options["s"][0]), int(options["s"][1])],
    )



def Build_MnasNet(model_name, override_params=None):
    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=0.2,
        data_format="channels_last",
        num_classes=1000,
        depth_multiplier=None,
        input_shape=(224, 224, 3),
        depth_divisor=8,
        min_depth=None,
        stem_size=32,
        normalize_input=True,
    )

    if model_name == "b1":
        blocks_args = [
            "r1_k3_s11_e1_i32_o16_noskip",
            "r3_k3_s22_e3_i16_o24",
            "r3_k5_s22_e3_i24_o40",
            "r3_k5_s22_e6_i40_o80",
            "r2_k3_s11_e6_i80_o96",
            "r4_k5_s22_e6_i96_o192",
            "r1_k3_s11_e6_i192_o320_noskip",
        ]

    elif model_name == "a1":
        blocks_args = [
            "r1_k3_s11_e1_i32_o16_noskip",
            "r2_k3_s22_e6_i16_o24",
            "r3_k5_s22_e3_i24_o40_se0.25",
            "r4_k3_s22_e6_i40_o80",
            "r2_k3_s11_e6_i80_o112_se0.25",
            "r3_k5_s22_e6_i112_o160_se0.25",
            "r1_k3_s11_e6_i160_o320",
        ]

    elif model_name == "small":
        blocks_args = [
            "r1_k3_s11_e1_i16_o8",
            "r1_k3_s22_e3_i8_o16",
            "r2_k3_s22_e6_i16_o16",
            "r4_k5_s22_e6_i16_o32_se0.25",
            "r3_k3_s11_e6_i32_o32_se0.25",
            "r3_k5_s22_e6_i32_o88_se0.25",
            "r1_k3_s11_e6_i88_o144",
        ]
        global_params = global_params._replace(dropout_rate=0.0, stem_size=8)

    elif model_name == "d1":
        blocks_args = [
            "r1_k3_s11_e9_i32_o24",
            "r3_k3_s22_e9_i24_o36",
            "r5_k3_s22_e9_i36_o48",
            "r4_k5_s22_e9_i48_o96",
            "r5_k7_s11_e3_i96_o96",
            "r3_k3_s22_e9_i96_o80",
            "r1_k7_s11_e6_i80_o320_noskip",
        ]

    elif model_name == "d1_320":
        blocks_args = [
            "r3_k5_s11_e6_i32_o24",
            "r4_k7_s22_e9_i24_o36",
            "r5_k5_s22_e9_i36_o48",
            "r5_k7_s22_e6_i48_o96",
            "r5_k3_s11_e9_i96_o144",
            "r5_k5_s22_e6_i144_o160",
            "r1_k7_s11_e9_i160_o320",
        ]

    else:
        raise NotImplementedError("model name is not pre-defined: %s" % model_name)

    if override_params:
        global_params = global_params._replace(**override_params)

    decoded_strings = [decode_block_string(s) for s in blocks_args]
    model = MnasNetModel(decoded_strings, global_params)
    return model