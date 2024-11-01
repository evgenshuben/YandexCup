#python3.10 main.py model.backbone_name='efficientnet_b0.ra4_e3600_r224_in1k'  experiment_number=11


# backbone start tune
#python3.10 main.py model.backbone_name='efficientnet_b0.ra4_e3600_r224_in1k'  experiment_number=12
#python3.10 main.py model.backbone_name='regnety_008_tv.tv2_in1k'  experiment_number=13
#python3.10 main.py model.backbone_name='regnety_040.ra3_in1k'  experiment_number=14

# backbone finetune
#/home/evgenshuben/Desktop/gitReps/YandexCup/outputs_val/run-6/data/model/best-model-epoch=22-max_secs=50.pt



python3.10 main.py model.backbone_name='hgnetv2_b4.ssld_stage2_ft_in1k' experiment_number=15 losses_weight.triplet=1.0 optimizer.lr=0.00001 pipeline.model_ckpt='/home/evgenshuben/Desktop/gitReps/YandexCup/outputs_val/run-6/data/model/best-model-epoch\=22-max_secs\=50.pt'