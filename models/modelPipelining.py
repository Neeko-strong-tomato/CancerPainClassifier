import os
import datetime
import communs.metrics as metrics
import communs.performanceAnalyser as analyser
import communs.trainingDataSaver as saver
import communs.trainingfunction as trainingfunction

class ModelPipeline:
    def __init__(self, model_name, training_fct, k_fold=False):
        self.model_name = model_name
        self.training_fct = training_fct
        self.k_fold = k_fold
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_id = f"{self.timestamp}_{self.model_name}"
        self.output_dir = os.path.join("results", self.session_id)
        os.makedirs(self.output_dir, exist_ok=True)
        self.model = None
        self.args = None
        self.logs = []

    def log(self, msg):
        print(msg)
        self.logs.append(msg)

    def save_logs(self):
        with open(os.path.join(self.output_dir, "log.txt"), "w") as f:
            f.write("\n".join(self.logs))

    def train(self, args):
        self.args = args
        self.log(f"Training started with args: {args}")
        
        self.model = self.training_fct(args)
        
        self.log("Training completed.")

        saver.save_model_weights(self.model, self.output_dir)

    def evaluate(self, args):
        
        results = analyser.evaluate_model(self.model, args)
        metrics.save_metrics(results, self.output_dir)

        saver.save_args(args, self.output_dir)

        analyser.plot_results(results, self.output_dir)

        self.save_logs()

    def get_model(self):
        return self.model