
Dice =  0.912594176634734 0.23467891562879228
Jaccard =  0.8912877629725225 0.24398452853122587
AP = 0.8661918087000763, std = 0.28510772258311445
AP2 = 0.8661918087000763
Finding best threshold to binarize...
100%|##############################################################################################################| 31/31 [00:30<00:00,  1.01it/s]
Best IOU: [0.86619181 0.28510772] at 0.42


 - resnext50_5folds




retrain 1 from 3, 1 from 4



## Results

Best IOU: [0.82425    0.32543346] at 0.6733333333333333 (Full DS) , 0.794 on LB   - Ternausnet out of the box
0.8235 at 0.5 (Full DS)


Best IOU: [0.831125   0.31617833] at 0.43333333333333335 (Fold 1)


LB 0.801
Changes: LR, is_deconv = True, albumentations, remove small element in submission

---------------------------------------------------------------------------------------------
ResNet34,
initial lr: 1-e4
loss: BCE+jacc*2
Reflect 101

epoch 69: Valid loss: 0.60517, jaccard: 0.47354, mean iou: 0.80575
Dice =  0.8556903526823452 0.3078951680634572
Jaccard =  0.8314135878585733 0.3135231270301588
AP = 0.7995, std = 0.348604001698202
Best IOU: [0.8015     0.34676757] at 0.5666666666666667

---------------------------------------------------------------------------------------------------
LL bce

Dice =  0.8580623330513839 0.30547681032272334
Jaccard =  0.833945467297123 0.3112523296066917
AP = 0.8025, std = 0.34712209667493077
AP2 = 0.8025
100%|############################################################| 31/31 [00:04<00:00,  7.70it/s]
Best IOU: [0.80925    0.34112232] at 0.6599999999999999

---------------------------------------------------------
LL lavash hinge

Dice =  0.8439879669758723 0.31892805478480807
Jaccard =  0.8195686812729808 0.3269080299365924
AP = 0.78825, std = 0.3637814419400748
AP2 = 0.78825
100%|######################################################################################################################| 31/31 [00:03<00:00,  7.87it/s]
Best IOU: [0.80025    0.35556285] at 0.5133333333333333

----------------------------------------------------------
LL BCE+ jackard, removed layer from conv1

Dice =  0.8580582126937302 0.30647425962199726
Jaccard =  0.8344651641044766 0.31219574537914846
AP = 0.8025, std = 0.3478056210011563
AP2 = 0.8025
100%|############################################################################################################| 31/31 [00:03<00:00,  8.02it/s]
Best IOU: [0.80425    0.34645626] at 0.6066666666666667

full dataset:

Dice =  0.8586968061085508 0.3030550035054379
Jaccard =  0.8340539102497554 0.3104207749035346
AP = 0.8006500000000001, std = 0.3502136169539957
AP2 = 0.8006500000000001
100%|############################################################################################################| 31/31 [00:20<00:00,  1.53it/s]
Best IOU: [0.800975   0.35028067] at 0.38
--------------------------------------------------
loss 0.1 bce + 0.9 lavasz

Dice =  0.8758269412342394 0.2858819735577914
Jaccard =  0.8526596376182437 0.2920842359885307
AP = 0.8225, std = 0.3306716649487827
AP2 = 0.8225
100%|###################################################################################################| 31/31 [00:03<00:00,  7.82it/s]
Best IOU: [0.825625  0.3256238] at 0.48666666666666664


----------------------------------------------------------------------------------------------------
Full val DS, 0.1 bce + lovash
ResNet34
Dice =  0.8899258166042203 0.26370581205515964
Jaccard =  0.8658187028609139 0.27297846743672044
AP = 0.834113457135589, std = 0.31657163478215083
AP2 = 0.834113457135589
100%|##########################################################################################################| 31/31 [00:17<00:00,  1.76it/s]
Best IOU: [0.83502925 0.31668857] at 0.58


-------------------------------------

Best IOU: [0.82113924 0.33027985] at 0.5933333333333333

Best IOU: [0.82797468 0.32357137] 

Best IOU: [0.83303797 0.31938966] at 0.5666666666666667 - regular 
]
Best IOU: [0.83126582 0.32122734] at 0.6333333333333333 - best model




AP = 0.8309590435003816, std = 0.3204321222223539
AP2 = 0.8309590435003816

--------------------------------------------------------
SE_ResNext 50 

Dice =  0.9028565799430888 0.24594858495001548
Jaccard =  0.87958110498131 0.2557415484279554
AP = 0.8504197405240397, std = 0.29843431318247376
AP2 = 0.8504197405240397
Finding best threshold to make mask null
100%|###########################################################################################################| 20/20 [00:19<00:00,  1.05it/s]
Best IOU: [0.85141185 0.29730395] at 20
Finding best threshold to binarize
100%|###########################################################################################################| 31/31 [00:29<00:00,  1.05it/s]
Best IOU: [0.85281099 0.29531939] at 0.4066666666666666


----------------------------------------------
Dice =  0.9103171966059468 0.23892232615378156
Jaccard =  0.8890446185384275 0.24741100385139306
AP = 0.8647672348003053, std = 0.2858888774894106
AP2 = 0.8647672348003053
Finding best threshold to make mask null
100%|###########################################################################################################| 20/20 [00:19<00:00,  1.03it/s]
Best IOU: [0.86509794 0.28555998] at 20
Finding best threshold to binarize
100%|###########################################################################################################| 31/31 [00:30<00:00,  1.03it/s]
Best IOU: [0.86547952 0.28556628] at 0.35333333333333333
