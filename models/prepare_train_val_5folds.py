from models.dataset import data_path
from sklearn.model_selection import StratifiedKFold, KFold
from skimage.io import imread
import os

# function used in startification by salt coverage
def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i :
            return i

def get_split(fold):

    train_path = os.path.join(data_path, 'train','images')
    ids = next(os.walk(train_path))[2]

    # to remove images with few pixels (<20)
    few_pixel_ids = ['052de39787.png', '07ac7e530f.png', '0bed5f2ace.png', '10b19bd1f8.png', '131ca4b83d.png', '13f448be07.png', '16589f0702.png', '1770691c51.png', '30e5cfec08.png', '3391f0befc.png', '346358e652.png', '38178b0ded.png', '3b05fe7a3f.png', '457d5edb4c.png', '45fb16d378.png', '46dd77ede5.png', '4766b51055.png', '48940ae0b0.png', '4b9862566c.png', '4be85a3110.png', '53eb356632.png', '568d119e66.png', '5ce554e890.png', '675cab373f.png', '6b4d65ac6a.png', '6c40978ddf.png', '6c45d80d1e.png', '7187f4c02c.png', '7242ab00b6.png', '739a9ab34a.png', '7a25e51377.png', '7f1d5f223c.png', '8138c79081.png', '81d37cb5fd.png', '8945b8916d.png', '8cbefc189e.png', '8f07ef8585.png', '921b60e76d.png', '956c91fe94.png', '96f26f6397.png', '9e9f3940a9.png', 'a446aa0ac8.png', 'ac931ace49.png', 'ad76d4cd21.png', 'afd0b385f2.png', 'b354751edd.png', 'b461a5b584.png', 'b7c1e2a377.png', 'b966734278.png', 'bf05a52a6b.png', 'c073b8930c.png', 'c1f92fd149.png', 'c37f7c51e9.png', 'c3a963f5e3.png', 'c78c89577c.png', 'c98504a8ab.png', 'c9c0097eff.png', 'ccdd1c542f.png', 'cfcffeda9e.png', 'd2522cfc93.png', 'd68c08baec.png', 'ee0be71990.png', 'efe3043924.png', 'f0228a88de.png', 'f5c25276d2.png', 'f87365827d.png', 'fbcd92f03f.png', 'fbf73cb975.png', 'fdb7d132be.png']
    ids = [id_ for id_ in ids if id_ not in few_pixel_ids]

    #empty_img_ids = ['05b69f83bf.png', '0d8ed16206.png', '10833853b3.png', '135ae076e9.png', '1a7f8bd454.png', '1b0d74b359.png', '1c0b2ceb2f.png', '1efe1909ed.png', '1f0b16aa13.png', '1f73caa937.png', '20ed65cbf8.png', '287b0f197f.png', '2fb6791298.png', '37df75f3a2.png', '3ee4de57f8.png', '3ff3881428.png', '40ccdfe09d.png', '423ae1a09c.png', '4f30a97219.png', '51870e8500.png', '573f9f58c4.png', '58789490d6.png', '590f7ae6e7.png', '5aa0015d15.png', '5edb37f5a8.png', '5ff89814f5.png', '6b95bc6c5f.png', '6f79e6d54b.png', '755c1e849f.png', '762f01c185.png', '7769e240f0.png', '808cbefd71.png', '8c1d0929a2.png', '8ee20f502e.png', '9260b4f758.png', '96049af037.png', '96d1d6138a.png', '97515a958d.png', '99909324ed.png', '9aa65d393a.png', 'a2b7af2907.png', 'a31e485287.png', 'a3e0a0c779.png', 'a48b9989ac.png', 'a536f382ec.png', 'a56e87840f.png', 'a8be31a3c1.png', 'a9e940dccd.png', 'a9fd8e2a06.png', 'aa97ecda8e.png', 'acb95dd7c9.png', 'b11110b854.png', 'b552fb0d9d.png', 'b637a7621a.png', 'b8c3ca0fab.png', 'b9bf0422a6.png', 'bedb558d15.png', 'c1c6a1ebad.png', 'c20069b110.png', 'c3589905df.png', 'c8404c2d4f.png', 'cc15d94784.png', 'd0244d6c38.png', 'd0e720b57b.png', 'd1665744c3.png', 'd2e14828d5.png', 'd6437d0c25.png', 'd8bed49320.png', 'd93d713c55.png', 'dcca025cc6.png', 'e0da89ce88.png', 'e51599adb5.png', 'e7da2d7800.png', 'e82421363e.png', 'ec542d0719.png', 'f0190fc4b4.png', 'f26e6cffd6.png', 'f2c869e655.png', 'f9fc7746fb.png', 'ff9d2e9ba7.png']
    #ids = [id_ for id_ in ids if id_ not in empty_img_ids]

    # will stratify by salt coverage
    sc = []
    masks_path = 'data/train/masks/'
    for id_ in ids:
        mask = imread(masks_path + id_, as_gray=True)
        sc.append((mask > 1).sum() / pow(mask.shape[0], 2))

    salt_class = list(map(cov_to_class, sc))


    X = range(0,len(ids))

    skf = StratifiedKFold(n_splits=5, shuffle=False, random_state=47)
    train_index, test_index  = list(skf.split(X, salt_class))[fold]

    # random folds without salt coverage startification
    # kf = KFold(n_splits=5, shuffle=False, random_state=47)
    #train_index, test_index  = list(kf.split(X))[fold]

    train_file_names = [ids[i][:-4] for i in train_index]
    val_file_names = [ids[i][:-4] for i in test_index]


    return train_file_names, val_file_names
