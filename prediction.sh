#cp runs/debug/model_best_0_118_8433.pt  data/models/ResNet34/model_0.pt
#rm data/predictions/ResNet34/OOF/*
python predictions.py --model_type IncV3 --fold 0
python evaluation.py