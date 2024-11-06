чтобы запустить пайплайн нужно создать директорию outputs_test

Установить зависимости в своем виртуальном окружении из [environment.yaml](environment.yaml)

Прокинуть в конфиг для гидры [config.yaml](configs/config.yaml) путь к датасету и урл сервиса млфлоу

```yaml
data_dir: /home/evgenshuben/Desktop/gitReps/YandexCup24/dataset
file_ext: npy
num_classes: 39535
mlflow:
  uri: урл с поднятым сервисом
  experiment_name: название 
  experiment_number: ${experiment_number}
```

# Обучение модели с 0
Чтобы обучать модели с нуля неоходимо сначала в гидре [config.yaml](configs/config.yaml) выбрать defaults
```yaml
defaults:
  - transforms/image  
  - transforms/augmentations
  - transforms/dataloader_classifier_collator_mixup_cutmix 
  - dataset/Classifier  
  - losses/cross_entropy
  - losses/triplet
  - model/backbone 
  - optimizer/adamW
```
далее по примеру run.sh прописать в терминале python3.10 main.py model.backbone_name='rexnetr_300.sw_in12k_ft_in1k'  experiment_number=19 указать номер экспа и название модели с timm (номер экспа должен быть уникальным в папке outputs_val)

После обучения модели необходимо дообучить второй раз, в конфиге [config.yaml](configs/config.yaml) указать дефаултс
```yaml
defaults:
  - transforms/image 
  - transforms/augmentations
  - transforms/dataloader_collator_default 
  - dataset/MetricLearning 
  - losses/cross_entropy
  - losses/triplet
  - model/backbone 
  - optimizer/adamW
```
Для рана с файнтюном вызвать команду (опять проставить уникальные номера экспов и для модели ее название)

python3.10 main.py model.backbone_name='efficientnet_b5.sw_in12k_ft_in1k' experiment_number=22 losses_weight.triplet=1.0 optimizer.lr=0.00001 pipeline.model_ckpt='/home/evgenshuben/Desktop/gitReps/YandexCup/outputs_val/run-20/data/model/best-model-epoch\=31-max_secs\=50.pt' pipeline.epochs=1




# Запуск ансамбля


Создать директорию outputs_val и распаковать в нее архив outputs_val.zip с яндекс диска 

В конфигах гидры (outputs_val/run-8/hydra/.hydra/config.yaml) необходимо прокинуть свои пути до трейн датасетов и указать урл поднятого сервиса mlflow


Запуск ансамбля на предобученных весах находится в ноутбуке notebooks/BackBones/Stacking.ipynb
