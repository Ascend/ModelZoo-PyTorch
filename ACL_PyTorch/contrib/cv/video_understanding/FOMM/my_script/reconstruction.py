import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from logger import Logger, Visualizer
import numpy as np
import imageio
from sync_batchnorm import DataParallelWithCallback


def reconstruction(config, generator, kp_detector, checkpoint, log_dir, dataset, data_dir="infer_out/", pre_data="pre_data/"):
    png_dir = os.path.join(log_dir, 'reconstruction/png')
    log_dir = os.path.join(log_dir, 'reconstruction')

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector)
    else:
        raise AttributeError("Checkpoint should be specified for mode='reconstruction'.")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    loss_list = []
    if torch.cuda.is_available():
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()

    if not data_dir.__contains__('/'):
        data_dir += "/"

    if not pre_data.__contains__('/'):
        pre_data += "/"
    kpdv_path = data_dir + "kpdv/"
    kpdj_path = data_dir + "kpdj/"
    kpsv_path = data_dir + "kpsv/"
    kpsj_path = data_dir + "kpsj/"
    source_path = pre_data + "source/"
    driving_path = pre_data + "driving/"
    out_path = data_dir + "out/"

    cnt = 0

    for it, x in tqdm(enumerate(dataloader)):
        if config['reconstruction_params']['num_videos'] is not None:
            if it > config['reconstruction_params']['num_videos']:
                break
        num = x['video'].shape[2]
        del x['video']
        file_num_file = np.load(pre_data + "frame_num.npy")
        file_num = file_num_file[it]
        if num != file_num:
            raise ValueError("{}:file num != num, num is {}, but file num is {}".format(it, num, file_num))
        with torch.no_grad():
            predictions = []
            visualizations = []
            for i in range(num):
                out = dict()
                kp_driving = dict()
                kp_source = dict()
                for j in range(5):
                    if j == 1:
                        continue
                    outi_path = out_path + str(cnt) + "_" + str(j) + ".npy"
                    outi = np.load(outi_path)
                    if j == 0:
                        out['mask'] = torch.from_numpy(outi).to(torch.float64)
                    elif j == 2:
                        out['occlusion_map'] = torch.from_numpy(outi).to(torch.float64)
                    elif j == 3:
                        out['deformed'] = torch.from_numpy(outi).to(torch.float64)
                    elif j == 4:
                        out['prediction'] = torch.from_numpy(outi).to(torch.float64)


                kp_driving_value_name = kpdv_path + str(cnt) + ".npy"
                kp_driving_jac_name = kpdj_path + str(cnt) + ".npy"
                kp_source_value_name = kpsv_path + str(cnt) + ".npy"
                kp_source_jac_name = kpsj_path + str(cnt) + ".npy"
                source_name = source_path + str(cnt) + ".npy"
                driving_name = driving_path + str(cnt) + ".npy"

                kp_driving_value = np.load(kp_driving_value_name)
                kp_driving_jac = np.load(kp_driving_jac_name)
                kp_source_value = np.load(kp_source_value_name)
                kp_source_jac = np.load(kp_source_jac_name)
                source = np.load(source_name)
                driving = np.load(driving_name)

                cnt += 1

                kp_driving['value'] = torch.from_numpy(kp_driving_value).to(torch.float64)
                kp_driving['jacobian'] = torch.from_numpy(kp_driving_jac).to(torch.float64)
                kp_source['value'] = torch.from_numpy(kp_source_value).to(torch.float64)
                kp_source['jacobian'] = torch.from_numpy(kp_source_jac).to(torch.float64)

                out['kp_source'] = kp_source
                out['kp_driving'] = kp_driving

                source = torch.from_numpy(source).to(torch.float64)
                driving = torch.from_numpy(driving).to(torch.float64)

                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

                visualization = Visualizer(**config['visualizer_params']).visualize(source=source, driving=driving, out=out)
                visualizations.append(visualization)

                loss_list.append(torch.abs(out['prediction'] - driving).mean().cpu().numpy())

            predictions = np.concatenate(predictions, axis=1)
            imageio.imsave(os.path.join(png_dir, x['name'][0] + '.png'), (255 * predictions).astype(np.uint8))

            image_name = x['name'][0] + config['reconstruction_params']['format']
            imageio.mimsave(os.path.join(log_dir, image_name), visualizations)

    print("Reconstruction loss: %s" % np.mean(loss_list))
