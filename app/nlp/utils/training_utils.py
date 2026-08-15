"""
Utilitários para treinamento de modelos.

Este módulo fornece classes e funções para facilitar o treinamento
dos modelos BERTimbau de forma padronizada.
"""

import os
import torch
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback
    # EarlyStoppingCallback removido para evitar conflito no Windows
)
from datasets import Dataset
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class MelhorEpocaCallback(TrainerCallback):
    """
    Guarda em memória os pesos da época com melhor métrica de validação e os
    restaura ao fim do treino.

    Substitui o `load_best_model_at_end` do Trainer, que exige gravar um
    checkpoint por época em disco (~440 MB cada). Como a busca de
    hiperparâmetros roda centenas de treinos, manter o melhor estado em RAM é
    muito mais barato.
    """

    def __init__(self, metrica='eval_f1', maior_melhor=True):
        self.metrica = metrica
        self.maior_melhor = maior_melhor
        self.melhor_valor = None
        self.melhor_epoca = None
        self.melhor_estado = None

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        if not metrics or self.metrica not in metrics or model is None:
            return
        valor = metrics[self.metrica]
        melhorou = (
            self.melhor_valor is None
            or (valor > self.melhor_valor if self.maior_melhor else valor < self.melhor_valor)
        )
        if melhorou:
            self.melhor_valor = valor
            self.melhor_epoca = state.epoch
            self.melhor_estado = {
                k: v.detach().to('cpu').clone() for k, v in model.state_dict().items()
            }

    def restaurar(self, model):
        """Devolve ao modelo os pesos da melhor época; informa se houve troca."""
        if self.melhor_estado is None:
            return False
        model.load_state_dict(self.melhor_estado)
        return True


def construir_otimizador(model, learning_rate, weight_decay, llrd=1.0):
    """
    Cria o AdamW com decaimento da taxa de aprendizado por camada (LLRD).

    Camadas mais baixas do BERT codificam traços linguísticos gerais e devem se
    mover menos que as camadas superiores, próximas da tarefa. Com `llrd` < 1,
    a taxa da camada *i* é `learning_rate * llrd^(n_camadas - i)`. Com
    `llrd = 1.0` o comportamento é o do AdamW padrão.

    Referência da técnica: ajuste fino discriminativo, usual em modelos
    pré-treinados de linguagem.
    """
    sem_decay = ('bias', 'LayerNorm.weight')

    def grupos_de(parametros, lr):
        com, sem = [], []
        for nome, p in parametros:
            if not p.requires_grad:
                continue
            (sem if any(s in nome for s in sem_decay) else com).append(p)
        saida = []
        if com:
            saida.append({'params': com, 'lr': lr, 'weight_decay': weight_decay})
        if sem:
            saida.append({'params': sem, 'lr': lr, 'weight_decay': 0.0})
        return saida

    if llrd >= 1.0:
        return torch.optim.AdamW(
            grupos_de(list(model.named_parameters()), learning_rate)
        )

    camadas = model.bert.encoder.layer
    n = len(camadas)

    grupos = []
    # Cabeça de classificação e pooler: taxa cheia
    cabeca = [(nome, p) for nome, p in model.named_parameters()
              if not nome.startswith('bert.encoder.layer') and not nome.startswith('bert.embeddings')]
    grupos += grupos_de(cabeca, learning_rate)

    # Encoder: taxa decai da última para a primeira camada
    for i, camada in enumerate(camadas):
        lr_camada = learning_rate * (llrd ** (n - 1 - i))
        grupos += grupos_de(list(camada.named_parameters()), lr_camada)

    # Embeddings: um degrau abaixo da primeira camada
    embeddings = list(model.bert.embeddings.named_parameters())
    grupos += grupos_de(embeddings, learning_rate * (llrd ** n))

    return torch.optim.AdamW(grupos)

class TrainingHelper:
    """Classe auxiliar para treinamento de modelos."""
    
    def __init__(
        self,
        task_name: str,
        model_name: str = "neuralmind/bert-base-portuguese-cased",
        output_base_dir: str = "./models"
    ):
        """
        Inicializa o helper de treinamento.
        
        Args:
            task_name: Nome da tarefa (AS, TOX, LI, TE)
            model_name: Nome do modelo base
            output_base_dir: Diretório base para salvar modelos
        """
        self.task_name = task_name
        self.model_name = model_name
        self.output_base_dir = output_base_dir
        
        # Cria diretório de saída se não existir
        os.makedirs(output_base_dir, exist_ok=True)
        
    def prepare_datasets(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: List[str],
        val_labels: List[int],
        test_texts: List[str],
        test_labels: List[int],
        tokenizer: BertTokenizer,
        max_length: int = 128
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Prepara os datasets para treinamento.
        
        Args:
            train_texts: Textos de treino
            train_labels: Labels de treino
            val_texts: Textos de validação
            val_labels: Labels de validação
            test_texts: Textos de teste
            test_labels: Labels de teste
            tokenizer: Tokenizador BERT
            max_length: Comprimento máximo das sequências
            
        Returns:
            Tuple com datasets tokenizados
        """
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=max_length
            )
        
        # Cria datasets
        train_dataset = Dataset.from_dict({
            'text': train_texts,
            'labels': train_labels
        })
        
        val_dataset = Dataset.from_dict({
            'text': val_texts,
            'labels': val_labels
        })
        
        test_dataset = Dataset.from_dict({
            'text': test_texts,
            'labels': test_labels
        })
        
        # Tokeniza
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        test_dataset = test_dataset.map(tokenize_function, batched=True)
        
        logger.info(f"Datasets preparados para {self.task_name}")
        logger.info(f"Treino: {len(train_dataset)} amostras")
        logger.info(f"Validação: {len(val_dataset)} amostras")
        logger.info(f"Teste: {len(test_dataset)} amostras")
        
        return train_dataset, val_dataset, test_dataset
    
    def get_training_args(
        self,
        output_dir: str,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 16,
        per_device_eval_batch_size: int = 16,
        learning_rate: float = 5e-5,
        warmup_steps: int = 0,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        logging_steps: int = 10,
        eval_steps: int = 500,
        save_steps: int = 500,
        evaluation_strategy: str = "steps",
        save_strategy: str = "steps",
        load_best_model_at_end: bool = True,
        metric_for_best_model: str = "eval_f1",
        greater_is_better: bool = True,
        fp16: bool = None,
        seed: int = 42
    ) -> TrainingArguments:
        """
        Cria argumentos de treinamento padronizados.

        O aquecimento é proporcional (`warmup_ratio`) e não um número fixo de
        passos: o corpus rende poucas centenas de passos por treino, e um
        warmup fixo de 500 passos consumiria quase toda a otimização.
        """
        # Precisão mista quando há GPU: aproximadamente dobra a vazão e permite
        # explorar um espaço de hiperparâmetros bem maior no mesmo tempo.
        if fp16 is None:
            fp16 = torch.cuda.is_available()

        return TrainingArguments(
            output_dir=output_dir,
            fp16=fp16,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            logging_dir=f"{output_dir}/logs",
            logging_steps=logging_steps,
            eval_steps=eval_steps,
            save_steps=save_steps,
            evaluation_strategy=evaluation_strategy,
            save_strategy=save_strategy,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            report_to=None,
            seed=seed
        )
    
    def compute_metrics(self, eval_pred):
        """
        Computa métricas durante o treinamento.

        A métrica principal ('f1') é o F1 macro, que trata as três classes com
        o mesmo peso — é a métrica reportada no artigo e a usada para
        selecionar hiperparâmetros. O F1 ponderado é mantido como informação
        adicional, mas não deve ser comparado com as baselines (que reportam
        macro).
        """
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro', zero_division=0
        )
        _, _, f1_weighted, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        accuracy = accuracy_score(labels, predictions)

        return {
            'accuracy': accuracy,
            'f1': f1,
            'f1_weighted': f1_weighted,
            'precision': precision,
            'recall': recall
        }
    
    def train_model(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        num_labels: int,
        training_args: TrainingArguments,
        model_path: Optional[str] = None,
        loss_type: str = "ce",
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
        llrd: float = 1.0,
        usar_melhor_epoca: bool = False
    ) -> Tuple[BertForSequenceClassification, Trainer]:
        """
        Treina o modelo.

        Args:
            llrd: fator de decaimento da taxa de aprendizado por camada
                (1.0 = AdamW padrão).
            usar_melhor_epoca: se True, restaura ao final os pesos da época com
                melhor `eval_f1` no conjunto passado como `val_dataset`. Esse
                conjunto deve ser um *dev* interno, separado do conjunto usado
                para reportar a métrica do fold — senão a métrica reportada fica
                otimista.
        """
        # Carrega o modelo
        if model_path and os.path.exists(model_path):
            model = BertForSequenceClassification.from_pretrained(
                model_path, num_labels=num_labels
            )
        else:
            model = BertForSequenceClassification.from_pretrained(
                self.model_name, num_labels=num_labels
            )
        
        class CustomTrainer(Trainer):
            def __init__(self, *args, **kwargs):
                self.loss_type = kwargs.pop("loss_type", "ce")
                self.focal_gamma = kwargs.pop("focal_gamma", 2.0)
                self.label_smoothing = kwargs.pop("label_smoothing", 0.0)
                self.class_weights = kwargs.pop("class_weights", None)
                super().__init__(*args, **kwargs)
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                # Os pesos de classe são criados na CPU; sem mover para o
                # dispositivo dos logits, o treino em GPU aborta.
                if self.class_weights is not None and self.class_weights.device != logits.device:
                    self.class_weights = self.class_weights.to(logits.device)
                if self.loss_type == "focal":
                    probs = F.softmax(logits, dim=-1)
                    pt = probs.gather(1, labels.unsqueeze(1)).squeeze()
                    ce = F.cross_entropy(
                        logits, labels,
                        weight=self.class_weights,
                        reduction="none",
                        label_smoothing=self.label_smoothing
                    )
                    loss = (ce * (1 - pt) ** self.focal_gamma).mean()
                else:
                    loss = F.cross_entropy(
                        logits, labels,
                        weight=self.class_weights,
                        label_smoothing=self.label_smoothing
                    )
                return (loss, outputs) if return_outputs else loss
        # Prepara pesos de classe se não fornecidos
        if class_weights is None:
            try:
                import numpy as np
                labels_arr = np.array(train_dataset["labels"])
                num_classes = num_labels
                counts = np.bincount(labels_arr, minlength=num_classes)
                total = counts.sum()
                weights = total / (num_classes * counts.clip(min=1))
                class_weights = torch.tensor(weights, dtype=torch.float)
            except Exception:
                class_weights = None
        callbacks = []
        melhor_epoca = None
        if usar_melhor_epoca:
            melhor_epoca = MelhorEpocaCallback(metrica='eval_f1', maior_melhor=True)
            callbacks.append(melhor_epoca)

        # Cria o trainer personalizado (SEM EarlyStoppingCallback)
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks or None,
            optimizers=(
                construir_otimizador(
                    model,
                    learning_rate=training_args.learning_rate,
                    weight_decay=training_args.weight_decay,
                    llrd=llrd
                ),
                None  # o scheduler é criado pelo Trainer sobre este otimizador
            ),
            loss_type=loss_type,
            focal_gamma=focal_gamma,
            label_smoothing=label_smoothing,
            class_weights=class_weights
        )

        logger.info(f"Iniciando treinamento para {self.task_name}")

        # Treina o modelo
        trainer.train()

        if melhor_epoca is not None and melhor_epoca.restaurar(model):
            logger.info(
                f"Pesos restaurados da época {melhor_epoca.melhor_epoca:.0f} "
                f"(eval_f1={melhor_epoca.melhor_valor:.4f})"
            )

        logger.info(f"Treinamento concluído para {self.task_name}")

        return model, trainer
    
    def save_model_with_metadata(
        self,
        model: BertForSequenceClassification,
        tokenizer: BertTokenizer,
        output_dir: str,
        training_args: TrainingArguments,
        metrics: Dict[str, float],
        additional_info: Optional[Dict[str, Any]] = None
    ):
        """
        Salva o modelo com metadados.
        """
        # Salva modelo e tokenizador
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Cria metadados
        metadata = {
            'task': self.task_name,
            'model_name': self.model_name,
            'training_date': datetime.now().isoformat(),
            'training_args': {
                'num_train_epochs': training_args.num_train_epochs,
                'per_device_train_batch_size': training_args.per_device_train_batch_size,
                'learning_rate': training_args.learning_rate,
                'warmup_steps': training_args.warmup_steps,
                'weight_decay': training_args.weight_decay
            },
            'final_metrics': metrics
        }
        
        if additional_info:
            metadata.update(additional_info)
        
        # Salva metadados. O encoding explícito é necessário: os metadados têm
        # texto acentuado e, sem ele, o Windows grava em cp1252 e o arquivo
        # falha na leitura como UTF-8.
        metadata_path = os.path.join(output_dir, 'training_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Modelo salvo em {output_dir} com metadados")
    
    def get_output_dir(self, experiment_name: Optional[str] = None) -> str:
        """
        Gera diretório de saída para o modelo.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if experiment_name:
            dir_name = f"{self.task_name}_{experiment_name}_{timestamp}"
        else:
            dir_name = f"{self.task_name}_{timestamp}"
        
        return os.path.join(self.output_base_dir, dir_name)


def create_training_config(
    task: str,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    max_length: int = 128
) -> Dict[str, Any]:
    """
    Cria configuração padrão para treinamento.
    """
    return {
        'task': task,
        'model_name': 'neuralmind/bert-base-portuguese-cased',
        'num_train_epochs': epochs,
        'per_device_train_batch_size': batch_size,
        'per_device_eval_batch_size': batch_size,
        'learning_rate': learning_rate,
        'max_length': max_length,
        'warmup_steps': 500,
        'weight_decay': 0.01,
        'logging_steps': 10,
        'eval_steps': 500,
        'save_steps': 500,
        'evaluation_strategy': 'steps',
        'save_strategy': 'steps',
        'load_best_model_at_end': True,
        'metric_for_best_model': 'eval_f1',
        'greater_is_better': True
    }
