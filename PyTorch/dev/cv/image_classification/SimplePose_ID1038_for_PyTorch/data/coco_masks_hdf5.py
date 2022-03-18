#!/usr/bin/env python
# coding:utf-8
#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
#
"""
Python script for generating the training and validation hdf5 data from MSCOCO dataset
"""
from pycocotools.coco import COCO
from scipy.spatial.distance import cdist
import numpy as np
import cv2
import os
import os.path
import h5py
import json
import time
import matplotlib.pyplot as plt

dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data/dataset/coco/link2coco2017'))

tr_anno_path = os.path.join(dataset_dir, "annotations/person_keypoints_train2017.json")  # 鍙彇keypoint鐨勬爣娉ㄤ俊鎭
tr_img_dir = os.path.join(dataset_dir, "train2017")

val_anno_path = os.path.join(dataset_dir, "annotations/person_keypoints_val2017.json")
val_img_dir = os.path.join(dataset_dir, "val2017")

datasets = [
    (val_anno_path, val_img_dir, "COCO_val"),  # it is important to have 'val' in validation dataset name,
    # look for 'val' below
    (tr_anno_path, tr_img_dir, "COCO")
]

tr_hdf5_path = os.path.join(dataset_dir, "coco_train_dataset384.h5")
val_hdf5_path = os.path.join(dataset_dir, "coco_val_dataset384.h5")

val_size = 100  # 5000  # size of validation set  璁剧疆鐨剉alidation subset鐨勫ぇ灏.銆鍓╀綑鐨剉al鏁版嵁灏嗛夊叆train鏁版嵁涓
image_size = 384  # 鐢ㄤ簬璁粌缃戠粶鏃讹紝璁惧畾鐨勮缁冮泦鍥剧墖鐨勭粺涓灏哄銆


def make_mask(img_dir, img_id, img_anns, coco):
    """Mask all unannotated people (including the crowd which has no keypoint annotation)"""
    # 瀵逛簬鏌愪竴寮犲浘鍍忓拰鍥惧儚涓墍鏈変汉鎴栬呬汉缇ょ殑鏍囨敞鍋氬鐞
    # mask miss 鍜屻mask all鐨勮В閲婏細
    # mask_all璁板綍浜嗕竴寮犲浘鍍忎笂鎵鏈変汉鐨刴ask(鍖呮嫭鍗曚釜浜哄拰涓缇や汉)锛屻鑰宮ask miss鏄负浜嗘帺鐩栨帀閭ｄ簺鏄汉锛屾湁segmentation浣嗘槸娌℃湁鏍囨敞keypoint
    # 闇瑕佹敞鎰忥紝mak miss鏄妸娌℃湁keypoint鐨勫尯鍩熷彉鎴0锛岃宮ask all鏄妸鎵鏈変汉鍖哄煙鍙樻垚锛戙傛渶鍚巑ask_miss鍙堜粠0,1 bool鍨嬪彉鍒0~255鐨剈int8
    # apply mask miss if p["num_keypoints"] <= 0 i.e. person is segmented but have no keypoints(joints)
    # "people who has little annotation(<5), who has little scale(<32*32) and who is so close to 'main_person'" are
    # not masked, they just can't be selected as main person of image, but they are still passed to the netwrok.
    # ----------------------------------------------------------------------------------- #
    #  (鎴戣涓鸿繖鏍峰仛鍙互浣垮緱缃戠粶瀵逛簬灏忎簬32澶у皬鐨勬垨鑰呰妭鐐瑰皯浜5鐨勪笉鏁忔劅锛屼娇寰楄缁冩椂鎶戝埗缃戠粶鍙互灞忚斀瀹冧滑锛
    # ----------------------------------------------------------------------------------- #
    # see:銆https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation/issues/43
    # About just "mask". It contains pixel segment borders. For my perception it never used in the algorithm,
    # may be it was created for some visualisation purposes.

    img_path = os.path.join(img_dir, "%012d.jpg" % img_id)

    if not os.path.exists(img_path):
        raise IOError("image path dose not exist: %s" % img_path)

    img = cv2.imread(img_path)
    h, w, c = img.shape
    # mask:銆https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation/issues/8#issuecomment-342977756
    mask_all = np.zeros((h, w), dtype=np.uint8)
    mask_miss = np.zeros((h, w), dtype=np.uint8)

    flag = 0
    for p in img_anns:
        seg = p["segmentation"]  # seg is just a boarder of an object, see annotation file

        if p["iscrowd"] == 1:  # the handel of crowd
            # segmentation鏍煎紡鍙栧喅浜庤繖涓疄渚嬫槸涓涓崟涓殑瀵硅薄锛堝嵆iscrowd=0锛屽皢浣跨敤polygons鏍煎紡锛夎繕鏄竴缁勫璞★紙鍗砳scrowd=1锛屽皢浣跨敤RLE鏍煎紡锛
            mask_crowd = coco.annToMask(p)

            temp = np.bitwise_and(mask_all, mask_crowd)  # 鎴戞劅瑙塼emp鏄箣鍓峬ask_all涓庡綋鍓峜rowded instances鐨刴ask鐨勪氦闆咺OU
            mask_crowd = mask_crowd - temp

            flag += 1
            continue
        else:
            mask = coco.annToMask(p)

        mask_all = np.bitwise_or(mask, mask_all)  # mask_all璁板綍浜嗕竴寮犲浘鍍忎笂鎵鏈変汉鐨刴ask
        # mask_all never used for anything except visualization !!!!
        if p["num_keypoints"] <= 0:
            mask_miss = np.bitwise_or(mask, mask_miss)

    if flag < 1:
        mask_miss = np.logical_not(mask_miss)
    elif flag == 1:
        # mask the few keypoint and crowded persons at the same time ! mask areas are 0 !
        mask_miss = np.logical_not(np.bitwise_or(mask_miss, mask_crowd))
        # mask all the persons including crowd, mask area are 1 !
        mask_all = np.bitwise_or(mask_all, mask_crowd)
    else:
        raise Exception("crowd segments > 1")  # 瀵逛竴涓尯鍩燂紝鍙兘瀛樺湪涓涓猻egment,涓嶅瓨鍦ㄤ竴涓尯鍩熷悓鏃跺睘浜庢煇涓や釜instances鐨勯儴鍒

    mask_miss = mask_miss.astype(np.uint8)
    mask_miss *= 255  # 淇濆瓨鐨勩mask_miss銆鐨勬暟鍊奸潪0鍗255

    mask_all = mask_all.astype(np.uint8)
    mask_all *= 255  # 淇濆瓨鐨勩mask_miss銆鐨勬暟鍊奸潪0鍗255
    # Mask miss is multiplied by the loss,
    # so masked areas are 0. (琚玬ask鐨勫尯鍩熸槸0) I.e. second mask is real mask miss. First mask (mask_all) is just for visuals.
    mask_concat = np.concatenate((mask_miss[:, :, np.newaxis], mask_all[:, :, np.newaxis]), axis=2)

    # # # # ------------ 娉ㄩ噴閮ㄥ垎浠ｇ爜鐢ㄦ潵鏄剧ずmask crowded instance  --------------
    # # # print('***************', mask_miss.min(), mask_miss.max())
    # plt.imshow(img[:,:,[2,1,0]])
    # plt.show()
    # plt.imshow(np.repeat(mask_concat[:, :, 1][:,:,np.newaxis], 3, axis=2))  # mask_all
    # plt.show()
    # plt.imshow(np.repeat(mask_concat[:, :, 0][:,:,np.newaxis], 3, axis=2))  # mask_miss
    # plt.show()
    # print('show')
    # # # -------------------------------------------------------------------

    return img, mask_concat


def process_image(image_rec, img_id, image_index, img_anns, dataset_type):
    # 閽堝澶勭悊鐨勫璞℃槸銆鏌愪竴寮爄d瀵瑰簲鐨刬mage銆鍙婅繖寮犲浘涓婃墍鏈変汉鐨勬爣娉

    numPeople = len(img_anns)
    h, w = image_rec['height'], image_rec['width']
    print("Image ID: ", img_id, '  ,', 'number of people: ', numPeople)

    all_persons = []

    for p in range(numPeople):

        pers = dict()  # 鐢ㄥ瓧鍏哥被鍨嬩繚瀛樻暟鎹

        person_center = [img_anns[p]["bbox"][0] + img_anns[p]["bbox"][2] / 2,  # 鏍囨敞鏍煎紡涓(x, y, w, h)
                         img_anns[p]["bbox"][1] + img_anns[p]["bbox"][3] / 2]

        pers["objpos"] = person_center  # objpos 浠ｈ〃鐨勬槸浜轰綋鐨勪腑蹇冧綅缃
        pers["bbox"] = img_anns[p]["bbox"]
        pers["segment_area"] = img_anns[p]["area"]
        pers["num_keypoints"] = img_anns[p]["num_keypoints"]

        anno = img_anns[p]["keypoints"]

        pers["joint"] = np.zeros((17, 3))
        for part in range(17):
            pers["joint"][part, 0] = anno[part * 3]  # x鍧愭爣锛 鍥犱负姣忎竴涓猵art鐨勪俊鎭湁(x, y, v) 3涓
            pers["joint"][part, 1] = anno[part * 3 + 1]  # y鍧愭爣锛屾敞鎰弜锛寉鍧愭爣鐨勫厛鍚庨『搴

            # visible/invisible
            # COCO - Each keypoint has a 0-indexed location x,y and a visibility flag v defined as v=0: not labeled
            # (in which case x=y=0), v=1: labeled but not visible, and v=2: labeled and visible.

            # OURS - # 3 never marked up in this dataset, 2 - not marked up in this person, 1 - marked and visible,
            # 0 - marked but invisible
            if anno[part * 3 + 2] == 2:  # +2銆瀵瑰簲visibility鐨勫
                pers["joint"][part, 2] = 1
            elif anno[part * 3 + 2] == 1:
                pers["joint"][part, 2] = 0
            else:
                pers["joint"][part, 2] = 2

        pers["scale_provided"] = img_anns[p]["bbox"][3] / image_size  # 姣忎竴涓猵erson鍗犳瘮
        # img_anns[p]["bbox"][3] 瀵瑰簲鐨勬槸浜轰綋妗嗙殑楂樺害h銆!銆

        all_persons.append(pers)

    main_persons = []
    prev_center = []

    """ 
    The idea of main person: each picture is feeded for each main person every epoch centered around this main 
    person( btw it is not working in michalfaber code, thats why quality of model is bit lower). 
    Secondary persons will not get such privilege, if they close to one of main persons they will be visible on crop,
     and heatmap/paf will be calculated, if they too far then ... bad luck, they never be machine learning star :)
    # https://github.com/anatolix/keras_Realtime_Multi-Person_Pose_Estimation/issues/14
    """
    # https://github.com/anatolix/keras_Realtime_Multi-Person_Pose_Estimation/issues/14

    for pers in all_persons:  # 鏈澶栧眰寰幆鏄墍鏈夌殑鍗曚釜鐨勪汉锛屼粠all person涓夊彇main銆person

        # skip this person if parts number is too low or if
        # segmentation area is too small
        if pers["num_keypoints"] < 5 or pers["segment_area"] < 32 * 32:
            #  we do not select the person, which is too small or has too few keypoints, as the main person
            # 鐢ㄤ簬灞呬腑鍥剧墖鐨刴ain person鏄敤鏉ヨ缁冪綉缁滅殑涓诲姏锛屾墍浠ュ叧閿偣鍜屼汉鐨勫ぇ灏忚鍚堢悊锛屽叧閿偣灏戠殑鍙兘绂诲叾浠杕ain person杩
            continue

        person_center = pers["objpos"]

        # skip this person if the distance to exiting person is too small
        flag = 0
        for pc in prev_center:  # prev_center 淇濆瓨浜唒erson銆center 浠ュ強浜轰綋妗嗛暱鍜屽涓殑鏈澶у
            a = np.expand_dims(pc[:2], axis=0)  # prev_center鏄竴涓潗鏍: (x, y)
            b = np.expand_dims(person_center, axis=0)
            dist = cdist(a, b)[0]  # by default, computing the euclidean distance
            # pc[2] 浠ｈ〃浜轰綋妗嗛暱鍜屽涓渶澶х殑閭ｄ竴杈,鍘熷绋嬪簭涓妸璺濈main person鐗瑰埆杩戯紝<0.3鐨刾erson涓嶅啀浣滀负涓嬩竴涓猰ain person
            # 鍥犱负杩欐牱娌℃湁蹇呰锛岀寰楀緢杩戯紝鍥剧墖瑁佸嚭鏉ョ殑閮ㄥ垎鍩烘湰涓鏍
            if dist < pc[2] * 0.3:
                flag = 1
                continue

        if flag == 1:
            continue

        main_persons.append(pers)  # 鑻ュ拰涔嬪墠宸茬粡瀛樺湪鐨勪汉璺濈涓嶆槸闈炲父杩戯紝骞朵笖鏍囨敞鐐逛笉灏戯紝鍒欐坊鍔犲綋鍓嶈繖涓汉锛岃涓烘槸涓涓猰ain person锛
        # 鍙兘鏄负浜嗛伩鍏嶇敓鎴愬樊寮傝緝灏忕殑璁粌鍥剧墖锛屽洜涓哄鏋滆窛绂诲緢杩戠殑璇濓紝渚濈劧浼氬寘鎷浉閭荤殑浜虹殑
        # main_persions鏄竴涓猯ist, pers鏄竴涓猟ic瀛楀吀锛屾帓搴忓湪绗竴鐨刴ain person灏嗕韩鏈変紭鍏堟潈
        prev_center.append(np.append(person_center, max(img_anns[p]["bbox"][2], img_anns[p]["bbox"][3])))
        # 淇濆瓨浜唒erson銆center 浠ュ強浜轰綋妗嗛暱鍜屽涓殑鏈澶у

    template = dict()
    template["dataset"] = dataset_type  # coco or coco_val

    if image_index < val_size and 'val' in dataset_type:  # notice: 'val' in 'COCOval'   >>>銆True
        isValidation = 1
    else:
        isValidation = 0

    template["isValidation"] = isValidation
    template["img_width"] = w
    template["img_height"] = h  # 杩欎釜鏄暣涓浘鍍忕殑w, h
    template["image_id"] = img_id  # 灏嗗寘鍚繖浜涗汉濮挎佹暟鎹殑image鐨刬d涔熶繚瀛樿捣鏉
    template["annolist_index"] = image_index
    template["img_path"] = '%012d.jpg' % img_id

    # 澶栭儴澶у惊鐜槸姣忎竴寮犲浘鐗囷紝鍐呴儴锛堜篃灏辨槸涓嬮潰杩欎釜锛夊惊鐜槸涓涓浘鐗囦腑鐨勬墍鏈塵ain_persons, 涔熷氨鏄姣忎竴涓猰ain_person閮戒細杞祦鍙樻垚鎺掑簭绗竴鐨勪汉锛
    # 灏嗕韩鏈夊浘鐗囧眳涓殑鐗规潈
    for p, person in enumerate(main_persons):  # p鏄痩ist鐨勭储寮曞簭鍙凤紝person鏄痙ic绫诲瀷鐨勪俊鎭唴瀹

        instance = template.copy()  # template is a dictionary type

        instance["objpos"] = [main_persons[p]["objpos"]]
        instance["joints"] = [main_persons[p]["joint"].tolist()]  # Return the array as a (possibly nested) list
        instance["scale_provided"] = [main_persons[p]["scale_provided"]]
        #  while training they scale main person to be approximately image size(368 pix in our case). But after
        #  it they do random scaling 0.6-1.1. So this is very logical network never learned libs(and PAFs) could be
        #  larger than half of image.
        lenOthers = 0

        for ot, operson in enumerate(all_persons):  # other person

            if person is operson:
                assert not "people_index" in instance, "several main persons? couldn't be"
                instance["people_index"] = ot
                continue

            if operson["num_keypoints"] == 0:
                continue

            instance["joints"].append(all_persons[ot]["joint"].tolist())
            instance["scale_provided"].append(all_persons[ot]["scale_provided"])
            instance["objpos"].append(all_persons[ot]["objpos"])

            lenOthers += 1

        assert "people_index" in instance, "No main person index"
        instance["numOtherPeople"] = lenOthers
        yield instance  # 甯︽湁yield鍏抽敭瀛楋紝鏄痝enerator
        #  闄や簡crowd鍜屽叧閿偣寰堝皯鐨勪汉浠ュ锛屾棦鎵撳寘浜唌ain person锛屼篃淇濆瓨浜嗗叾浠栭潪main person锛屽浜庝竴涓猧nstance锛屾瘡娆″彧鏈変竴涓紭鍏堟潈main person


def writeImage(grp, img_grp, data, img, mask_miss, count, image_id, mask_grp=None):
    """
    Write hdf5 files
    :param grp: annotation hdf5 group
    :param img_grp: image hdf5 group
    :param data: annotation handled
    :param img: image returned by mask_mask()
    :param mask_miss: mask returned by mask_mask()
    :param count:
    :param image_id: image index
    :param mask_grp: mask hdf5 group
    :return: nothing
    """
    serializable_meta = data
    serializable_meta['count'] = count

    nop = data['numOtherPeople']

    assert len(serializable_meta['joints']) == 1 + nop, [len(serializable_meta['joints']), 1 + nop]
    assert len(serializable_meta['scale_provided']) == 1 + nop, [len(serializable_meta['scale_provided']), 1 + nop]
    assert len(serializable_meta['objpos']) == 1 + nop, [len(serializable_meta['objpos']), 1 + nop]

    img_key = "%012d" % image_id
    if not img_key in img_grp:

        if mask_grp is None:  # 涓轰簡鍏煎MPII娌℃湁mask鐨勬儏褰
            img_and_mask = np.concatenate((img, mask_miss[..., None]), axis=2)
            # create_dataset 杩斿洖鍒涘缓鐨刪df5瀵硅薄(姝ゅ涓篿mg_ds)锛屽苟涓旀瀵硅薄琚坊鍔犲埌img_key(鑻ataset name涓嶄负None)涓
            img_ds = img_grp.create_dataset(img_key, data=img_and_mask, chunks=None)
        else:
            # _, img_bin = cv2.imencode(".jpg", img)  # encode compress, we do not need it actually, delete cv2.imencode
            # _, img_mask = cv2.imencode(".png", mask_miss) # data= img_bin, data = img_mask
            img_ds1 = img_grp.create_dataset(img_key, data=img, chunks=None)
            img_ds2 = mask_grp.create_dataset(img_key, data=mask_miss, chunks=None)

    key = '%07d' % count
    required = {'image': img_key, 'joints': serializable_meta['joints'], 'objpos': serializable_meta['objpos'],
                'scale_provided': serializable_meta['scale_provided']}
    ds = grp.create_dataset(key, data=json.dumps(required), chunks=None)
    ds.attrs['meta'] = json.dumps(serializable_meta)

    print('Writing sample %d' % count)


def process():
    tr_h5 = h5py.File(tr_hdf5_path, 'w')
    tr_grp = tr_h5.create_group("dataset")
    tr_write_count = 0
    tr_grp_img = tr_h5.create_group("images")
    tr_grp_mask = tr_h5.create_group("masks")  # in fact, is mask_concat rather than mask_miss  NOTICE !!!

    val_h5 = h5py.File(val_hdf5_path, 'w')
    val_grp = val_h5.create_group("dataset")
    val_write_count = 0
    val_grp_img = val_h5.create_group("images")
    val_grp_mask = val_h5.create_group("masks")

    for _, ds in enumerate(datasets):
        # datasets = [(val_anno_path, val_img_dir, "COCO_val"),(tr_anno_path, tr_img_dir, "COCO")]
        anno_path = ds[0]
        img_dir = ds[1]
        dataset_type = ds[2]

        coco = COCO(anno_path)
        ids = list(coco.imgs.keys())

        for image_index, img_id in enumerate(ids):
            ann_ids = coco.getAnnIds(imgIds=img_id)
            img_anns = coco.loadAnns(ann_ids)
            image_rec = coco.imgs[img_id]

            img = None
            mask_miss = None
            cached_img_id = None

            for data in process_image(image_rec, img_id, image_index, img_anns, dataset_type):
                # 鐢眕rocess_image涓殑val_size鎺у埗楠岃瘉闆嗙殑澶у皬

                if cached_img_id != data['image_id']:
                    assert img_id == data['image_id']
                    cached_img_id = data['image_id']
                    img, mask_miss = make_mask(img_dir, cached_img_id, img_anns, coco)

                if data['isValidation']:  # 鏍规嵁 isValidation 鏍囧織绗︾‘瀹氭槸鍚︿綔涓簐al
                    writeImage(val_grp, val_grp_img, data, img, mask_miss, val_write_count, cached_img_id, val_grp_mask)
                    val_write_count += 1
                else:
                    writeImage(tr_grp, tr_grp_img, data, img, mask_miss, tr_write_count, cached_img_id, tr_grp_mask)
                    tr_write_count += 1
    tr_h5.close()
    val_h5.close()
    return tr_write_count, val_write_count


if __name__ == '__main__':
    start_time = time.time()
    tr_sample, val_sample = process()
    end_time = time.time()
    print('************************** \n')
    print('coco mask data process finished! consuming time: %.3f min' % ((end_time - start_time) / 60))
    print('the size of train sample is: ', tr_sample)
    print('the size of val sample is: ', val_sample)
    # 澶х害闇瑕佸鐞30 min
