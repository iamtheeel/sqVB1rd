import torch
from Pruning_Engine.pruning_engine import pruning_engine
from Model import MobileNetV3

class optimiser:
    def __init__(self, model, shalPruneRatio, midPruneRatio, deepPruneRatio):
        self.model = model

        self.pruner = pruning_engine(pruning_method="L1norm",individual = True)

        self.shalPruneRatio = shalPruneRatio
        self.midPruneRatio = midPruneRatio
        self.deepPruneRatio = deepPruneRatio

        self.shallow_layer_idx = self.model.conv2d_layers[:self.model.shaLayEnd]
        self.shallow_bn_layers_idx = self.model.bn_layers[:self.model.shaLayEnd]
        self.mid_layer_idx = self.model.conv2d_layers[self.model.shaLayEnd-1:self.model.midLayEnd]
        self.mid_bn_layers_idx = self.model.bn_layers[self.model.shaLayEnd-1:self.model.midLayEnd]
        self.deep_layer_idx = self.model.conv2d_layers[self.model.midLayEnd-1:]
        self.deep_bn_layers_idx = self.model.bn_layers[self.model.midLayEnd-1:]

    def prune(self):

        ## From HW6
        """
        Pruning deep CNN Layers
        """
        print(f"Prune it too deep: {self.deepPruneRatio}" )
        # i = 0
        # list = self.model.named_children()
        # print(list)
        # for item in self.model.named_children():
        #     print(f"item: {item})
        # for name, module in self.model.named_modules():
        #     i = i + 1
        #     if isinstance(module, torch.nn.Conv2d):
        #         print(f"Convilutional Layer: {name}, layer:{i}, inCh: {module.in_channels}, outCh: {module.out_channels}")
        #         self.pruner.set_pruning_ratio(self.deepPruneRatio)
        #         pruned_layer = module
        #         self.pruner.set_layer(pruned_layer, main_layer=True)
        #         remove_filter_idx = self.pruner.get_remove_filter_idx()["current_layer"]
        #         self.pretrain_model.features[self.deep_layer_idx[index]] = self.pruner.remove_filter_by_index(remove_filter_idx)

        for index in range(len(self.deep_layer_idx) - 1):
            print(f"Layer {self.deep_layer_idx[index]}")
            self.pruner.set_pruning_ratio(self.deepPruneRatio)
            #prune the conv2d layers filter
            pruned_layer = self.model.features[self.deep_layer_idx[index]]
            self.pruner.set_layer(pruned_layer,main_layer=True)
            remove_filter_idx = self.pruner.get_remove_filter_idx()["current_layer"]
            self.model.features[self.deep_layer_idx[index]] = self.pruner.remove_filter_by_index(remove_filter_idx)

            #prune the conv2d layers filter kernal
            pruned_layer = self.model.features[self.deep_layer_idx[index+1]]
            self.pruner.set_layer(pruned_layer)
            self.model.features[self.deep_layer_idx[index+1]] = self.pruner.remove_kernel_by_index(remove_filter_idx)

            #prune the Batch mormalized layers layers filter kernal
            pruned_layer = self.model.features[self.deep_bn_layers_idx[index]]
            self.pruner.set_layer(pruned_layer)
            self.model.features[self.deep_bn_layers_idx[index]] = self.pruner.remove_Bn(remove_filter_idx)
        print(f"Prune done: " )


        return self.model