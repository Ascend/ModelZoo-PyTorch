# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from distutils.log import info
import os
import glob
import numpy as np
import io
import six
from itertools import chain
import cv2
import json
import sys
import multiprocessing
sys.path.append(os.getcwd())
from prepostprocess import PrePostProcess
from utils import rect_iou, center_error


deviceid = 0


class OTB(object):
    r"""`OTB <http://cvlab.hanyang.ac.kr/tracker_benchmark/>`_ Datasets.

    Publication:
        ``Object Tracking Benchmark``, Y. Wu, J. Lim and M.-H. Yang, IEEE TPAMI 2015.

    Args:
        root_dir (string): Root directory of dataset where sequence
            folders exist.
        version (integer or string): Specify the benchmark version, specify as one of
            ``2013``, ``2015``, ``tb50`` and ``tb100``.
    """
    __otb13_seqs = ['Basketball', 'Bolt', 'Boy', 'Car4', 'CarDark',
                    'CarScale', 'Coke', 'Couple', 'Crossing', 'David',
                    'David2', 'David3', 'Deer', 'Dog1', 'Doll', 'Dudek',
                    'FaceOcc1', 'FaceOcc2', 'Fish', 'FleetFace',
                    'Football', 'Football1', 'Freeman1', 'Freeman3',
                    'Freeman4', 'Girl', 'Ironman', 'Jogging', 'Jumping',
                    'Lemming', 'Liquor', 'Matrix', 'Mhyang', 'MotorRolling',
                    'MountainBike', 'Shaking', 'Singer1', 'Singer2',
                    'Skating1', 'Skiing', 'Soccer', 'Subway', 'Suv',
                    'Sylvester', 'Tiger1', 'Tiger2', 'Trellis', 'Walking',
                    'Walking2', 'Woman']

    __tb50_seqs = ['Basketball', 'Biker', 'Bird1', 'BlurBody', 'BlurCar2',
                   'BlurFace', 'BlurOwl', 'Bolt', 'Box', 'Car1', 'Car4',
                   'CarDark', 'CarScale', 'ClifBar', 'Couple', 'Crowds',
                   'David', 'Deer', 'Diving', 'DragonBaby', 'Dudek',
                   'Football', 'Freeman4', 'Girl', 'Human3', 'Human4',
                   'Human6', 'Human9', 'Ironman', 'Jump', 'Jumping',
                   'Liquor', 'Matrix', 'MotorRolling', 'Panda', 'RedTeam',
                   'Shaking', 'Singer2', 'Skating1', 'Skating2', 'Skiing',
                   'Soccer', 'Surfer', 'Sylvester', 'Tiger2', 'Trellis',
                   'Walking', 'Walking2', 'Woman']

    __tb100_seqs = ['Bird2', 'BlurCar1', 'BlurCar3', 'BlurCar4', 'Board',
                    'Bolt2', 'Boy', 'Car2', 'Car24', 'Coke', 'Coupon',
                    'Crossing', 'Dancer', 'Dancer2', 'David2', 'David3',
                    'Dog', 'Dog1', 'Doll', 'FaceOcc1', 'FaceOcc2', 'Fish',
                    'FleetFace', 'Football1', 'Freeman1', 'Freeman3',
                    'Girl2', 'Gym', 'Human2', 'Human5', 'Human7', 'Human8',
                    'Jogging', 'KiteSurf', 'Lemming', 'Man', 'Mhyang',
                    'MountainBike', 'Rubik', 'Singer1', 'Skater',
                    'Skater2', 'Subway', 'Suv', 'Tiger1', 'Toy', 'Trans',
                    'Twinnings', 'Vase'] + __tb50_seqs

    __otb15_seqs = __tb100_seqs

    __version_dict = {
        2013: __otb13_seqs,
        2015: __otb15_seqs,
        'otb2013': __otb13_seqs,
        'otb2015': __otb15_seqs,
        'tb50': __tb50_seqs,
        'tb100': __tb100_seqs}

    def __init__(self, root_dir, version=2015):
        super(OTB, self).__init__()
        assert version in self.__version_dict

        self.root_dir = root_dir
        self.version = version
        self._check_integrity(root_dir, version)
        valid_seqs = self.__version_dict[version]
        self.anno_files = sorted(list(chain.from_iterable(glob.glob(
            os.path.join(root_dir, s, 'groundtruth*.txt')) for s in valid_seqs)))
        # remove empty annotation files
        # (e.g. groundtruth_rect.1.txt of Human4)
        self.anno_files = self._filter_files(self.anno_files)
        self.seq_dirs = [os.path.dirname(f) for f in self.anno_files]
        self.seq_names = [os.path.basename(d) for d in self.seq_dirs]
        # rename repeated sequence names
        # (e.g. Jogging and Skating2)
        self.seq_names = self._rename_seqs(self.seq_names)

    def __getitem__(self, index):
        r"""
        Args:
            index (integer or string): Index or name of a sequence.

        Returns:
            tuple: (img_files, anno), where ``img_files`` is a list of
                file names and ``anno`` is a N x 4 (rectangles) numpy array.
        """
        if isinstance(index, six.string_types):
            if not index in self.seq_names:
                raise Exception('Sequence {} not found.'.format(index))
            index = self.seq_names.index(index)

        img_files = sorted(glob.glob(
            os.path.join(self.seq_dirs[index], 'img/*.jpg')))

        # special sequences
        seq_name = self.seq_names[index]
        if seq_name.lower() == 'david':
            img_files = img_files[300 - 1:770]
        elif seq_name.lower() == 'football1':
            img_files = img_files[:74]
        elif seq_name.lower() == 'freeman3':
            img_files = img_files[:460]
        elif seq_name.lower() == 'freeman4':
            img_files = img_files[:283]
        elif seq_name.lower() == 'diving':
            img_files = img_files[:215]

        # to deal with different delimeters
        with open(self.anno_files[index], 'r') as f:
            anno = np.loadtxt(io.StringIO(f.read().replace(',', ' ')))
        assert len(img_files) == len(anno)
        assert anno.shape[1] == 4

        return img_files, anno

    def __len__(self):
        return len(self.seq_names)

    def _filter_files(self, filenames):
        filtered_files = []
        for filename in filenames:
            with open(filename, 'r') as f:
                if f.read().strip() == '':
                    print('Warning: %s is empty.' % filename)
                else:
                    filtered_files.append(filename)

        return filtered_files

    def _rename_seqs(self, seq_names):
        # in case some sequences may have multiple targets
        renamed_seqs = []
        for i, seq_name in enumerate(seq_names):
            if seq_names.count(seq_name) == 1:
                renamed_seqs.append(seq_name)
            else:
                ind = seq_names[:i + 1].count(seq_name)
                renamed_seqs.append('%s.%d' % (seq_name, ind))

        return renamed_seqs

    def _check_integrity(self, root_dir, version):
        assert version in self.__version_dict
        seq_names = self.__version_dict[version]

        if os.path.isdir(root_dir) and len(os.listdir(root_dir)) > 0:
            # check each sequence folder
            for seq_name in seq_names:
                seq_dir = os.path.join(root_dir, seq_name)
                if not os.path.isdir(seq_dir):
                    print('Warning: sequence %s not exists.' % seq_name)
        else:
            # dataset not exists
            raise Exception('Dataset not found or corrupted. ' +
                            'You can use download=True to download it.')


class ExperimentOTB(object):
    """Experiment pipeline and evaluation toolkit for OTB dataset.

        Args:
            root_dir (string): Root directory of OTB dataset.
            version (integer or string): Specify the benchmark version, specify as one of
                ``2013``, ``2015``, ``tb50`` and ``tb100``. Default is ``2015``.
            result_dir (string, optional): Directory for storing tracking
                results. Default is ``./results``.
            report_dir (string, optional): Directory for storing performance
                evaluation results. Default is ``./reports``.
    """
    def __init__(self, root_dir, version=2015,
                 result_dir='results', report_dir='reports'):
        super(ExperimentOTB, self).__init__()
        self.dataset = OTB(root_dir, version)
        self.result_dir = os.path.join(result_dir, 'OTB' + str(version))
        self.report_dir = os.path.join(report_dir, 'OTB' + str(version))
        # as nbins_iou increases, the success score
        # converges to the average overlap (AO)
        self.nbins_iou = 21
        self.nbins_ce = 51

    def getlendataset(self):
        return len(self.dataset)

    def run(self, savepath, infopath, idx):
        # get the seq_name and information of files
        img_files, anno = self.dataset[idx]
        seq_name = self.dataset.seq_names[idx]
        # generate directory for current seq
        savepath = savepath + "/" + str(idx)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        infopath = infopath + "/" + str(idx) + ".info"
        # skip if result exist
        record_file = os.path.join(self.result_dir, 'siamfc', '%s.txt' % seq_name)
        if os.path.exists(record_file):
            print('Found results of %s, skipping' % seq_name)
            return
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = anno[0, :]  # x，y, w, h
        times = np.zeros(frame_num)

        prepostpro = PrePostProcess()
        for f, img_file in enumerate(img_files):
            print(seq_name + "  %s/%s" %(f, frame_num))
            img = cv2.imread(img_file, cv2.IMREAD_COLOR)
            if f == 0:
                # Pre-process and generate bin
                exemplar_path = prepostpro.cropexemplar(img, anno[0, :], savepath, img_file)
                # get_info
                with open(infopath, 'w') as file1:
                    content = ' '.join([str(0), '.'+exemplar_path, str(127), str(127)])
                    file1.write(content)
                    file1.write('\n')
                # infer
                # os.system('./benchmark.%s -model_type=vision -device_id=%d -batch_size=1 '
                #           '-om_path=./om/exemplar_bs1.om -input_text_path=%s '
                #           '-input_width=127 -input_height=127 -output_binary=True -useDvpp=False >/dev/null 2>&1'
                #           % (arch, deviceid, infopath))
                os.system('python3.7 ./ais_infer/ais_infer.py  --model ./om/exemplar_bs1.om '
                 '--input pre_dataset/%s --device_id 0 -o ./OTB100 --outfmt BIN --output_prefix %s >/dev/null 2>&1'
                 %(idx, seq_name))
                # the exemplar has a result of 3*256*6*6 tensor
                # read tensor from bin
                # filename = img_file.replace('/', '-').split('.')[0] + '_1.bin'
                filename = 'sample_id_0_output_0.bin'
                filename = 'OTB100/'+ seq_name+ '/' + filename
                exemplar_feature = prepostpro.file2tensor(filename, (3, 256, 6, 6))
                os.system('rm -rf ./pre_dataset/%s/%s' %(idx,img_file.replace('/', '-').replace('.jpg', '.bin')))
            else:
                # Pre-process and generate bin
                search_path = prepostpro.cropsearch(img, savepath, img_file)
                # get_info
                with open(infopath, 'w') as file2:
                    content = ' '.join([str(0), '.'+search_path, str(255), str(255)])
                    file2.write(content)
                    file2.write('\n')
                # infer
                # os.system('./benchmark.%s -model_type=vision -device_id=%d -batch_size=1 '
                #           '-om_path=./om/search_bs1.om -input_text_path=%s '
                #           '-input_width=255 -input_height=255 -output_binary=True -useDvpp=False >/dev/null 2>&1'
                #           % (arch, deviceid, infopath))
                os.system('python3.7 ./ais_infer/ais_infer.py  --model ./om/search_bs1.om '
                    '--input pre_dataset/%s --device_id 0 -o ./OTB100 --outfmt BIN --output_prefix %s >/dev/null 2>&1'
                    %(idx, seq_name))
                # the exemplar has a result of 1*768*22*22 tensor
                # read tensor from bin
                # filename = img_file.replace('/', '-').split('.')[0] + '_1.bin'
                # filename = 'result/dumpOutput_device' + str(deviceid) + '/' + filename
                filename = 'sample_id_0_output_0.bin'
                filename = 'OTB100/'+ seq_name+ '/' + filename
                search_feature = prepostpro.file2tensor(filename, (1, 768, 22, 22))
                # Post-process
                boxes[f, :] = prepostpro.postprocess(search_feature, exemplar_feature)
                times[f] = 1
                os.system('rm -rf ./pre_dataset/%s/%s' %(idx, img_file.replace('/', '-').replace('.jpg', '.bin')))

        assert len(boxes) == len(anno)
        # record results
        self._record(record_file, boxes, times)
        # delete useless data to save space
        os.system('rm -rf %s/*' % savepath)
        print("Results of %s finished!" % seq_name)

    def report(self, tracker_names):

        assert isinstance(tracker_names, (list, tuple))  # ‘SiamFC’

        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, tracker_names[0])

        if not os.path.isdir(report_dir):
            os.makedirs(report_dir)

        report_file = os.path.join(report_dir, 'performance.json')

        performance = {}
        for name in tracker_names:
            print('Evaluating', name)
            seq_num = len(self.dataset)
            succ_curve = np.zeros((seq_num, self.nbins_iou))
            prec_curve = np.zeros((seq_num, self.nbins_ce))
            speeds = np.zeros(seq_num)
            #
            performance.update({name: {'overall': {}, 'seq_wise': {}}})

            for s, (_, anno) in enumerate(self.dataset):

                seq_name = self.dataset.seq_names[s]

                record_file = os.path.join(self.result_dir, name, '%s.txt' % seq_name)

                boxes = np.loadtxt(record_file, delimiter=',')

                boxes[0] = anno[0]

                assert len(boxes) == len(anno)

                ious, center_errors = self._calc_metrics(boxes, anno)

                succ_curve[s], prec_curve[s] = self._calc_curves(ious, center_errors)

                # calculate average tracking speed
                time_file = os.path.join(self.result_dir, name, 'times/%s_time.txt' % seq_name)

                if os.path.isfile(time_file):
                    times = np.loadtxt(time_file)
                    times = times[times > 0]
                    if len(times) > 0:
                        speeds[s] = np.mean(1. / times)
                # store sequence-wise performance
                performance[name]['seq_wise'].update({seq_name: {
                    'success_curve': succ_curve[s].tolist(),
                    'precision_curve': prec_curve[s].tolist(),
                    'success_score': np.mean(succ_curve[s]),
                    'precision_score': prec_curve[s][20],
                    'success_rate': succ_curve[s][self.nbins_iou // 2],
                    'speed_fps': speeds[s] if speeds[s] > 0 else -1}})

            succ_curve = np.mean(succ_curve, axis=0)
            prec_curve = np.mean(prec_curve, axis=0)
            succ_score = np.mean(succ_curve)
            prec_score = prec_curve[20]
            succ_rate = succ_curve[self.nbins_iou // 2]
            if np.count_nonzero(speeds) > 0:
                avg_speed = np.sum(speeds) / np.count_nonzero(speeds)
            else:
                avg_speed = -1

            # store overall performance
            performance[name]['overall'].update({
                'success_curve': succ_curve.tolist(),
                'precision_curve': prec_curve.tolist(),
                'success_score': succ_score,
                'precision_score': prec_score,
                'success_rate': succ_rate,
                'speed_fps': avg_speed})
            # print('prec_score:%s --succ_score:%s --succ_rate:%s' % (prec_score,succ_score,succ_rate))
        # report the performance
        with open(report_file, 'w') as f:
            json.dump(performance, f, indent=4)

        return prec_score, succ_score, succ_rate

    def _record(self, record_file, boxes, times):
        # record bounding boxes
        record_dir = os.path.dirname(record_file)
        if not os.path.isdir(record_dir):
            os.makedirs(record_dir)
        np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')

        # print('  Results recorded at', record_file)

        # record running times
        time_dir = os.path.join(record_dir, 'times')
        if not os.path.isdir(time_dir):
            os.makedirs(time_dir)
        time_file = os.path.join(time_dir, os.path.basename(
            record_file).replace('.txt', '_time.txt'))
        np.savetxt(time_file, times, fmt='%.8f')

    def _calc_metrics(self, boxes, anno):
        # can be modified by children classes
        ious = rect_iou(boxes, anno)
        center_errors = center_error(boxes, anno)
        return ious, center_errors

    def _calc_curves(self, ious, center_errors):
        ious = np.asarray(ious, float)[:, np.newaxis]
        center_errors = np.asarray(center_errors, float)[:, np.newaxis]

        thr_iou = np.linspace(0, 1, self.nbins_iou)[np.newaxis, :]
        thr_ce = np.arange(0, self.nbins_ce)[np.newaxis, :]

        bin_iou = np.greater(ious, thr_iou)
        bin_ce = np.less_equal(center_errors, thr_ce)

        succ_curve = np.mean(bin_iou, axis=0)
        prec_curve = np.mean(bin_ce, axis=0)

        return succ_curve, prec_curve


if __name__ == "__main__":
    data_path = sys.argv[1]
    save_path = sys.argv[2]
    info_path = sys.argv[3]
    deviceid = int(sys.argv[4])
    os.system('rm -rf %s' % save_path)
    os.system('rm -rf %s' % info_path)
    os.system('rm -rf ./result/dumpOutput_device%d' % deviceid)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(info_path):
        os.makedirs(info_path)
    e = ExperimentOTB(data_path, version=2015)
    totallen = e.getlendataset()
    pool = multiprocessing.Pool(processes=12)
    for i in range(totallen):
        pool.apply_async(e.run, (save_path, info_path, i, ))
    pool.close()
    pool.join()
    prec_score, succ_score, succ_rate = e.report(['siamfc'])
    ss = '-prec_score:%.3f -succ_score:%.3f -succ_rate:%.3f' % (float(prec_score), float(succ_score), float(succ_rate))
    print("====accuracy data====")
    print(ss)

