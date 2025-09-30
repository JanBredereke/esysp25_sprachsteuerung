import torch
from torch.nn import Module
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat
import torch.nn as nn


class SuperModel(Module):
    def __init__(self):
        super().__init__()

    def __iter__(self):
        return self._modules.values().__iter__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Custom forward method for the model. If not implemented by subclass, the layers are trained sequentially.

        :param x The incoming vector to process by the network layers
        """
        for layer in self:
            x = layer.forward(x)
        return x


class SuperModelSeq:
    def __init__(self):
        super().__init__()


class DefaultModel(SuperModel):
    def __init__(self, in_features, number_of_categories):
        super().__init__()
        self.layer1 = qnn.QuantLinear(
            in_features,
            in_features * 2,
            bias=False,
            weight_quant=Int8WeightPerTensorFloat,
            weight_bit_width=3,
            return_quant_tensor=True,
        )
        self.layer2 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)
        self.layer3 = qnn.QuantLinear(
            in_features * 2,
            in_features * 2,
            weight_quant=Int8WeightPerTensorFloat,
            bias=False,
            weight_bit_width=3,
            return_quant_tensor=True,
        ),
        self.layer4 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)
        self.layer5 = qnn.QuantLinear(
            in_features * 2,
            number_of_categories,
            weight_quant=Int8WeightPerTensorFloat,
            bias=False,
            weight_bit_width=3,
            return_quant_tensor=False,
        )
        self.layer6 = nn.BatchNorm1d(number_of_categories)


class Model2(SuperModel):
    def __init__(self, in_features, number_of_categories):
        super().__init__()
        self.layer1 = qnn.QuantLinear(
            in_features,
            in_features * 2,
            bias=False,
            weight_quant=Int8WeightPerTensorFloat,
            weight_bit_width=3,
            return_quant_tensor=True,
        )
        self.layer2 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)
        self.layer3 = qnn.QuantLinear(
            in_features * 2,
            in_features * 2,
            weight_quant=Int8WeightPerTensorFloat,
            bias=False,
            weight_bit_width=3,
            return_quant_tensor=True,
        )
        self.layer4 = qnn.QuantDropout(0.2, return_quant_tensor=True)
        self.lin = qnn.QuantLinear(
            in_features * 2,
            in_features * 2,
            weight_quant=Int8WeightPerTensorFloat,
            bias=False,
            weight_bit_width=3,
            return_quant_tensor=True,
        )
        self.layer5 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)
        self.layer6 = qnn.QuantDropout(0.2, return_quant_tensor=True)
        self.layer7 = qnn.QuantMaxPool1d(4, return_quant_tensor=True)
        self.layer8 = qnn.QuantLinear(
            int(in_features / 2),
            number_of_categories,
            weight_quant=Int8WeightPerTensorFloat,
            bias=False,
            weight_bit_width=3,
            return_quant_tensor=True,
        )
        self.norm = nn.BatchNorm1d(number_of_categories)


# geht nicht :(
class ExampleEditSeq(SuperModelSeq):
    def __init__(self, in_features, number_of_categories):
        super().__init__()
        self.model = nn.Sequential(
            qnn.QuantLinear(in_features, in_features * 2, bias=False, weight_bit_width=3, return_quant_tensor=True),
            qnn.BatchNorm1dToQuantScaleBias(in_features * 2, return_quant_tensor=True),
            qnn.QuantDropout(0.5, return_quant_tensor=True),
            qnn.QuantReLU(bit_width=4, return_quant_tensor=True),
            qnn.QuantLinear(in_features * 2, in_features * 2, bias=False, weight_bit_width=3, return_quant_tensor=True),
            qnn.BatchNorm1dToQuantScaleBias(in_features * 2, return_quant_tensor=True),
            qnn.QuantDropout(0.5, return_quant_tensor=True),
            qnn.QuantReLU(bit_width=4, return_quant_tensor=True),
            qnn.QuantLinear(in_features * 2, in_features * 2, bias=False, weight_bit_width=3, return_quant_tensor=True),
            qnn.BatchNorm1dToQuantScaleBias(in_features * 2, return_quant_tensor=True),
            qnn.QuantDropout(0.5, return_quant_tensor=True),
            qnn.QuantReLU(bit_width=4, return_quant_tensor=True),
            qnn.QuantLinear(in_features * 2, number_of_categories, bias=False, weight_bit_width=3,
                            return_quant_tensor=True)
        )


# geht nicht :(
class ExampleEdit(SuperModel):
    def __init__(self, in_features, number_of_categories):
        super().__init__()
        self.layer1 = qnn.QuantLinear(in_features, in_features * 2, bias=False, weight_bit_width=3,
                                      return_quant_tensor=True)
        self.layer2 = qnn.BatchNorm1dToQuantScaleBias(in_features * 2, return_quant_tensor=True)
        self.layer3 = qnn.QuantDropout(0.5, return_quant_tensor=True)
        self.layer4 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)
        self.layer5 = qnn.QuantLinear(in_features * 2, in_features * 2, bias=False, weight_bit_width=3,
                                      return_quant_tensor=True)
        self.layer6 = qnn.BatchNorm1dToQuantScaleBias(in_features * 2, return_quant_tensor=True)
        self.layer7 = qnn.QuantDropout(0.5, return_quant_tensor=True)
        self.layer8 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)
        self.layer9 = qnn.QuantLinear(in_features * 2, in_features * 2, bias=False, weight_bit_width=3,
                                      return_quant_tensor=True)
        self.layer10 = qnn.BatchNorm1dToQuantScaleBias(in_features * 2, return_quant_tensor=True)
        self.layer11 = qnn.QuantDropout(0.5, return_quant_tensor=True)
        self.layer12 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)
        self.layer13 = qnn.QuantLinear(in_features * 2, number_of_categories, bias=False, weight_bit_width=3,
                                       return_quant_tensor=True)


class ExampleSeq(SuperModelSeq):
    def __init__(self, in_features, number_of_categories):
        super().__init__()
        self.model = nn.Sequential(
            qnn.QuantLinear(in_features, in_features * 2, bias=True, weight_bit_width=3),
            nn.BatchNorm1d(in_features * 2),
            nn.Dropout(0.5),
            qnn.QuantReLU(bit_width=4),
            qnn.QuantLinear(in_features * 2, in_features * 2, bias=True, weight_bit_width=3),
            nn.BatchNorm1d(in_features * 2),
            nn.Dropout(0.5),
            qnn.QuantReLU(bit_width=4),
            qnn.QuantLinear(in_features * 2, in_features * 2, bias=True, weight_bit_width=3),
            nn.BatchNorm1d(in_features * 2),
            nn.Dropout(0.5),
            qnn.QuantReLU(bit_width=4),
            qnn.QuantLinear(in_features * 2, number_of_categories, bias=True, weight_bit_width=3)
        )


class MyExampleSeq(SuperModelSeq):
    def __init__(self, in_features, number_of_categories):
        super().__init__()
        self.model = nn.Sequential(
            qnn.QuantLinear(in_features, in_features * 2, bias=True, weight_bit_width=3),
            nn.BatchNorm1d(in_features * 2),
            nn.Dropout(0.5),
            qnn.QuantReLU(bit_width=4),
            qnn.QuantLinear(in_features * 2, in_features * 2, bias=True, weight_bit_width=3),
            nn.BatchNorm1d(in_features * 2),
            nn.Dropout(0.5),
            qnn.QuantReLU(bit_width=4),
            qnn.QuantLinear(in_features * 2, in_features * 2, bias=True, weight_bit_width=3),
            nn.BatchNorm1d(in_features * 2),
            nn.Dropout(0.5),
            qnn.QuantReLU(bit_width=4),
            qnn.QuantLinear(in_features * 2, number_of_categories, bias=True, weight_bit_width=3)
        )


class Example(SuperModel):
    def __init__(self, in_features, number_of_categories):
        super().__init__()
        self.layer1 = qnn.QuantLinear(in_features, in_features * 2, bias=True, weight_bit_width=3)
        self.layer2 = nn.BatchNorm1d(in_features * 2)
        self.layer3 = nn.Dropout(0.5)
        self.layer4 = qnn.QuantReLU(bit_width=4)
        self.layer5 = qnn.QuantLinear(in_features * 2, in_features * 2, bias=True, weight_bit_width=3)
        self.layer6 = nn.BatchNorm1d(in_features * 2)
        self.layer7 = nn.Dropout(0.5)
        self.layer8 = qnn.QuantReLU(bit_width=4)
        self.layer9 = qnn.QuantLinear(in_features * 2, in_features * 2, bias=True, weight_bit_width=3)
        self.layer10 = nn.BatchNorm1d(in_features * 2)
        self.layer11 = nn.Dropout(0.5)
        self.layer12 = qnn.QuantReLU(bit_width=4)
        self.layer13 = qnn.QuantLinear(in_features * 2, number_of_categories, bias=True, weight_bit_width=3)
