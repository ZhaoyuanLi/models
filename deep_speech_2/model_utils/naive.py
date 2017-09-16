from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.v2 as paddle


def conv_bn_layer(input, filter_size, num_channels_in, num_channels_out, stride,
                  padding, act):
    """Convolution layer with batch normalization.

    :param input: Input layer.
    :type input: LayerOutput
    :param filter_size: The x dimension of a filter kernel. Or input a tuple for
                        two image dimension.
    :type filter_size: int|tuple|list
    :param num_channels_in: Number of input channels.
    :type num_channels_in: int
    :type num_channels_out: Number of output channels.
    :type num_channels_in: out
    :param padding: The x dimension of the padding. Or input a tuple for two
                    image dimension.
    :type padding: int|tuple|list
    :param act: Activation type.
    :type act: BaseActivation
    :return: Batch norm layer after convolution layer.
    :rtype: LayerOutput
    """
    conv_layer = paddle.layer.img_conv(
        input=input,
        filter_size=filter_size,
        num_channels=num_channels_in,
        num_filters=num_channels_out,
        stride=stride,
        padding=padding,
        act=paddle.activation.Linear(),
        bias_attr=False)
    return paddle.layer.batch_norm(input=conv_layer, act=act)


def feed_forward_layer(input, size,act):
    """Feed Forward Layer

    :param input: Input layer.
    :type input: LayerOutput
    :param size: Number of cells in the layer.
    :type size: int
    :param act: Activation type.
    :type act: BaseActivation

    :return: Feed Forward Layer
    :rtype: LayerOutput
    """
    ff_layer = paddle.layer.fc(
        input=input,
        size=size,
        act=act)
    return ff_layer

def feed_forward_group(input, size, num_stacks):
    """Feed Forward group with layers.

    :param input: Input layer.
    :type input: LayerOutput
    :param size: Number of neurons in each layer
    :type size: int
    :param num_stacks: Number of stacked feed forward layers.
    :type num_stacks: int
    :return: Output layer of the convolution group.
    :rtype: LayerOutput
    """
    ff = feed_forward_layer(
        input = input,
        size = size,
        act = paddle.activation.Relu())
    for i in range(num_stacks - 1):
        ff = feed_forward_layer(
            input = ff,
            size = size,
            act = paddle.activation.Relu())
    return ff

def conv_group(input, num_stacks):
    """Convolution group with stacked convolution layers.

    :param input: Input layer.
    :type input: LayerOutput
    :param num_stacks: Number of stacked convolution layers.
    :type num_stacks: int
    :return: Output layer of the convolution group.
    :rtype: LayerOutput
    """
    conv = conv_bn_layer(
        input=input,
        filter_size=(11, 41),
        num_channels_in=1,
        num_channels_out=32,
        stride=(3, 2),
        padding=(5, 20),
        act=paddle.activation.BRelu())
    for i in xrange(num_stacks - 1):
        conv = conv_bn_layer(
            input=conv,
            filter_size=(11, 21),
            num_channels_in=32,
            num_channels_out=32,
            stride=(1, 2),
            padding=(5, 10),
            act=paddle.activation.BRelu())
    output_num_channels = 32
    output_height = 160 // pow(2, num_stacks) + 1
    return conv, output_num_channels, output_height





def naive_network(audio_data,
                           text_data,
                           dict_size,
                           num_conv_layers=2,
                           num_ff_layers = 2,
                           ff_size = 256):
    """The DeepSpeech2 network structure.

    :param audio_data: Audio spectrogram data layer.
    :type audio_data: LayerOutput
    :param text_data: Transcription text data layer.
    :type text_data: LayerOutput
    :param dict_size: Dictionary size for tokenized transcription.
    :type dict_size: int
    :param num_conv_layers: Number of stacking convolution layers.
    :type num_conv_layers: int
    :param num_ff_layers: Number of stacking FF layers.
    :type num_rnn_layers: int
    :param ff_size: FF layer size (number of FF cells).
    :type ff_size: int
    :return: A tuple of an output unnormalized log probability layer (
             before softmax) and a ctc cost layer.
    :rtype: tuple of LayerOutput
    """
    # convolution group
    conv_group_output, conv_group_num_channels, conv_group_height = conv_group(
        input=audio_data, num_stacks=num_conv_layers)
    # convert data form convolution feature map to sequence of vectors
    conv2seq = paddle.layer.block_expand(
        input=conv_group_output,
        num_channels=conv_group_num_channels,
        stride_x=1,
        stride_y=1,
        block_x=1,
        block_y=conv_group_height)
    # feed forward group
    ff = feed_forward_group(
            input = conv2seq,
            size = ff_size,
            num_stacks = num_ff_layers)

    fc = paddle.layer.fc(
        input=ff,
        size=dict_size + 1,
        act=paddle.activation.Linear(),
        bias_attr=True)
    # probability distribution with softmax
    log_probs = paddle.layer.mixed(
        input=paddle.layer.identity_projection(input=fc),
        act=paddle.activation.Softmax())
    # ctc cost
    ctc_loss = paddle.layer.warp_ctc(
        input=fc,
        label=text_data,
        size=dict_size + 1,
        blank=dict_size,
        norm_by_times=True)
    return log_probs, ctc_loss
