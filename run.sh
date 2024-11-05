#python3.10 main.py model.backbone_name='efficientnet_b0.ra4_e3600_r224_in1k'  experiment_number=11


# backbone start tune
#python3.10 main.py model.backbone_name='efficientnet_b0.ra4_e3600_r224_in1k'  experiment_number=12
#python3.10 main.py model.backbone_name='regnety_008_tv.tv2_in1k'  experiment_number=13
#python3.10 main.py model.backbone_name='regnety_040.ra3_in1k'  experiment_number=14

# backbone finetune
#/home/evgenshuben/Desktop/gitReps/YandexCup/outputs_val/run-6/data/model/best-model-epoch=22-max_secs=50.pt



#python3.10 main.py model.backbone_name='hgnetv2_b4.ssld_stage2_ft_in1k' experiment_number=15 losses_weight.triplet=1.0 optimizer.lr=0.00001 pipeline.model_ckpt='/home/evgenshuben/Desktop/gitReps/YandexCup/outputs_val/run-6/data/model/best-model-epoch\=22-max_secs\=50.pt'


#python3.10 main.py model.backbone_name='tf_efficientnet_b4.ns_jft_in1k' experiment_number=16 losses_weight.triplet=1.0 optimizer.lr=0.00001 pipeline.model_ckpt='/home/evgenshuben/Desktop/gitReps/YandexCup/outputs_val/run-4/data/model/best-model-epoch\=14-max_secs\=50.pt' pipeline.epochs=5



#python3.10 main.py model.backbone_name='edgenext_base.in21k_ft_in1k'  experiment_number=16

#python3.10 main.py model.backbone_name='edgenext_base.in21k_ft_in1k' experiment_number=17 losses_weight.triplet=1.0 optimizer.lr=0.00001 pipeline.model_ckpt='/home/evgenshuben/Desktop/gitReps/YandexCup/outputs_val/run-16/data/model/best-model-epoch\=16-max_secs\=50.pt' pipeline.epochs=5





#python3.10 main.py model.backbone_name='tf_efficientnet_b4.ns_jft_in1k' experiment_number=18 losses_weight.triplet=1.0 optimizer.lr=0.00001 pipeline.model_ckpt='/home/evgenshuben/Desktop/gitReps/YandexCup/outputs_val/run-4/data/model/best-model-epoch\=14-max_secs\=50.pt' pipeline.epochs=5






#python3.10 main.py model.backbone_name='rexnetr_300.sw_in12k_ft_in1k'  experiment_number=19
#python3.10 main.py model.backbone_name='efficientnet_b5.sw_in12k_ft_in1k'  experiment_number=20
#python3.10 main.py model.backbone_name='hgnetv2_b5.ssld_stage2_ft_in1k'  experiment_number=21
#python3.10 main.py model.backbone_name='tf_efficientnet_b6.ns_jft_in1k'  experiment_number=22
#python3.10 main.py model.backbone_name='regnetz_d8_evos.ch_in1k'  experiment_number=23 # тяжелая модель, ее в конец






# fintune 19, 20 exp
#python3.10 main.py model.backbone_name='rexnetr_300.sw_in12k_ft_in1k' experiment_number=21 losses_weight.triplet=1.0 optimizer.lr=0.00001 pipeline.model_ckpt='/home/evgenshuben/Desktop/gitReps/YandexCup/outputs_val/run-19/data/model/best-model-epoch\=38-max_secs\=50.pt' pipeline.epochs=1
#python3.10 main.py model.backbone_name='efficientnet_b5.sw_in12k_ft_in1k' experiment_number=22 losses_weight.triplet=1.0 optimizer.lr=0.00001 pipeline.model_ckpt='/home/evgenshuben/Desktop/gitReps/YandexCup/outputs_val/run-20/data/model/best-model-epoch\=31-max_secs\=50.pt' pipeline.epochs=1



python3.10 main.py model.backbone_name='hgnetv2_b5.ssld_stage2_ft_in1k'  experiment_number=23
python3.10 main.py model.backbone_name='tf_efficientnet_b6.ns_jft_in1k'  experiment_number=24
python3.10 main.py model.backbone_name='regnetz_d8_evos.ch_in1k'  experiment_number=25 # тяжелая модель, ее в конец


# rexnetr_300.sw_in12k_ft_in1k
# efficientnet_b5.sw_in12k_ft_in1k
# hgnetv2_b5.ssld_stage2_ft_in1k
# tf_efficientnet_b6.ns_jft_in1k





