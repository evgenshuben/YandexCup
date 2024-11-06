# Запуск ансамбля

чтобы запустить пайплайн нужно создать директорию outputs_test

Создать директорию outputs_val и распаковать в нее архив outputs_val.zip с яндекс диска 

В конфигах гидры (outputs_val/run-8/hydra/.hydra/config.yaml) необходимо прокинуть свои пути до трейн датасетов и указать урл поднятого сервиса mlflow

```yaml
data_dir: /home/evgenshuben/Desktop/gitReps/YandexCup24/dataset
file_ext: npy
num_classes: 39535
mlflow:
  uri: http://127.0.0.1:8080
  experiment_name: TestRun
  experiment_number: ${experiment_number}
```


Запуск ансамбля на предобученных весах находится в ноутбуке notebooks/BackBones/Stacking.ipynb



# Обучение модели с 0
Чтобы обучать модели с нуля неоходимо сначала в гидре выбрать defaults
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

После обучения модели необходимо дообучить второй раз, в конфиге указать дефаултс
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

