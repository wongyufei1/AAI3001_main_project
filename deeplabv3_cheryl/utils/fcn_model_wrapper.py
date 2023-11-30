import torch.nn as nn
from utils.model_wrapper import SemanticModelWrapper

class FCNModelWrapper(SemanticModelWrapper):
    def __init__(self, model, n_classes, device="cpu", weights=None, optimizer=None, epochs=None):
        # Assuming n_classes is the number of output classes for your FCN mode
        # Call the superclass (SemanticModelWrapper) constructor
        super().__init__(model, device, n_classes, weights, optimizer, epochs)
    
    def config_model(self, out_classes, weights):
        #configure last layer of model 
        filters_of_last_layer = self.model.classifier[4].in_channels
        filters_of_last_layer_aux = self.model.aux_classifier[4].in_channels
        self.model.classifier[4] = nn.Conv2d(filters_of_last_layer,out_classes,kernel_size=(1,1),stride=(1,1))
        self.model.aux_classifier[4] = nn.Conv2d(filters_of_last_layer_aux,out_classes, kernel_size=(1,1),stride=(1,1))
        
        if weights is not None:
            self.model.load_state_dict(weights)

        self.model.to(self.device)
