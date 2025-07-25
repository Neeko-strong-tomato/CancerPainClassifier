import communs.metrics
import communs.performanceAnalyser
import communs.trainingDataSaver
import communs.trainingfunction

class modelPipelining:

    def __init__(self, model_name, training_fct, K_fold=False):
        self.model = model_name
        self.training_fct = training_fct

    def train(self, args):
        self.training_fct(args)