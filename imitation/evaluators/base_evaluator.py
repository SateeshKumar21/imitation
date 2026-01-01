class MultiEvals:

    def __init__(self, eval_config, trainer, *args, **kwargs):
        self.eval_config = eval_config

        self.eval_list = []
        for evaluator in eval_config.evaluator_configs:
            self.eval_list.append(evaluator.evaluator(evaluator, trainer, **kwargs))

    def evaluate(self, policy):
        info = {'key_to_modality': {}}
        for evaluator in self.eval_list:
            eval_dict = evaluator.evaluate(policy)
            for k, v in eval_dict.items():
                if k=='key_to_modality':
                    info['key_to_modality'].update(v)
                    continue
                if k in info:
                    raise ValueError(f"Key {k} already exists in info dictionary. Please ensure unique keys across evaluators.")
                else:
                    info[k] = v
        return info